"""Application services shared by the HTTP API and future workers."""

from __future__ import annotations

import hashlib
from typing import Any
from uuid import UUID

from .chunking import chunk_text, normalize_text
from .config import Settings
from .db import (
    delete_document,
    find_document_by_hash,
    get_audit_logs,
    get_document,
    get_document_chunks,
    get_system_stats,
    list_documents,
    record_audit_log,
    save_document,
    search_chunks,
    search_chunks_hybrid,
)
from .embeddings import get_embedder
from .llm import LLMClient
from .quality import diagnose_text
from .reranker import get_reranker, rerank_chunks
from .standardization import (
    detect_and_mask_pii,
    evaluate_enterprise_quality,
    standardize_administrative_terms,
)
from .structured_quality import evaluate_structured_dataset


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ingest_text(
    settings: Settings,
    *,
    name: str,
    text: str,
    source_type: str,
    mime_type: str | None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    # The input is not archived as a separate object. The database stores the
    # normalized text; with PII masking disabled that text can still be unmasked.
    if settings.ingest_auto_mask_pii:
        processed_text, pii_candidates = detect_and_mask_pii(text, mask=True)
    else:
        processed_text, pii_candidates = text, detect_and_mask_pii(text, mask=False)[1]
    suggested_text, term_candidates = standardize_administrative_terms(processed_text)
    std_text = suggested_text if settings.ingest_apply_local_term_replacements else processed_text
    normalized = normalize_text(std_text)
    if not normalized:
        raise ValueError("추출된 텍스트가 비어 있습니다.")

    content_hash = _content_hash(normalized)
    existing = find_document_by_hash(settings, name, content_hash)
    quality = evaluate_enterprise_quality(
        text,
        name,
        auto_mask_pii=settings.ingest_auto_mask_pii,
        apply_local_term_replacements=settings.ingest_apply_local_term_replacements,
    )
    if existing:
        return {
            "document_id": existing["id"],
            "name": existing["source_name"],
            "chunk_count": existing["chunk_count"],
            "duplicate": True,
            "quality": existing["quality_report"],
        }

    text_chunks = chunk_text(normalized, settings.chunk_size, settings.chunk_overlap)
    if not text_chunks:
        raise ValueError("문서를 청크로 나누지 못했습니다.")

    chunks = [
        {
            "index": chunk.index,
            "text": chunk.text,
            "metadata": {"char_start": chunk.start, "char_end": chunk.end},
        }
        for chunk in text_chunks
    ]
    embeddings = get_embedder(settings).embed_documents([chunk["text"] for chunk in chunks])
    processing_metadata = {
        "pipeline_version": "rag-vllm-ingestion-v2",
        "input_content_retained_separately": False,
        "stored_content_is_derived": True,
        "stored_content_masked": settings.ingest_auto_mask_pii,
        "content_transformations": [
            "NFKC_and_whitespace_normalization",
            *(["heuristic_pii_masking"] if settings.ingest_auto_mask_pii else []),
            *(["local_term_replacements"] if settings.ingest_apply_local_term_replacements else []),
        ],
        "pii_detection": "heuristic_pattern_match",
        "pii_candidates_detected": len(pii_candidates),
        "pii_masking_applied": settings.ingest_auto_mask_pii,
        "term_candidates_detected": term_candidates,
        "local_term_replacements_applied": settings.ingest_apply_local_term_replacements,
        "standardization_status": "local_candidates_pending_review",
        "official_standardization_verified": False,
    }
    persisted_metadata = {**metadata, "rag_vllm_processing": processing_metadata}
    document_id, duplicate, chunk_count = save_document(
        settings,
        source_name=name,
        source_type=source_type,
        mime_type=mime_type,
        content_hash=content_hash,
        content=normalized,
        metadata=persisted_metadata,
        quality_report=quality,
        chunks=chunks,
        embeddings=embeddings,
    )
    return {
        "document_id": document_id,
        "name": name,
        "chunk_count": chunk_count,
        "duplicate": duplicate,
        "quality": quality,
    }


def build_context(hits: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[{index}] source={hit['source_name']} chunk={hit['chunk_index']}\n{hit['text']}"
        for index, hit in enumerate(hits, start=1)
    )


def query_rag(
    settings: Settings,
    *,
    question: str,
    top_k: int,
    document_id: UUID | None,
    min_quality_score: int | None,
    use_llm: bool,
    search_mode: str = "hybrid",
    department: str | None = None,
    max_security_level: int | None = None,
    client_ip: str | None = None,
) -> dict[str, Any]:
    question = question.strip()
    if not question:
        raise ValueError("질문이 비어 있습니다.")

    embedder = get_embedder(settings)
    query_embedding = embedder.embed_query(question)

    # Stage 1: Candidate retrieval
    candidate_k = max(15, top_k * 3)
    if search_mode == "hybrid":
        raw_hits = search_chunks_hybrid(
            settings,
            query_embedding,
            question,
            top_k=candidate_k,
            document_id=document_id,
            min_quality_score=min_quality_score,
            department=department,
            max_security_level=max_security_level,
        )
    else:
        raw_hits = search_chunks(
            settings,
            query_embedding,
            top_k=candidate_k,
            document_id=document_id,
            min_quality_score=min_quality_score,
            department=department,
            max_security_level=max_security_level,
        )

    # Stage 2: Cross-Encoder / High-Precision Reranker
    reranker = get_reranker(settings)
    hits = rerank_chunks(question, raw_hits, top_k=top_k, reranker=reranker)

    context = build_context(hits)
    llm = LLMClient(settings)
    answer = llm.complete(question, context) if use_llm and hits else None

    # Confidence and Hallucination Risk Assessment
    confidence_score = 0.0
    hallucination_risk = "low"
    if hits:
        avg_score = sum(float(h.get("score") or h.get("rrf_score", 0.5)) for h in hits) / len(hits)
        confidence_score = round(min(1.0, max(0.0, avg_score)), 2)
        if answer:
            has_citations = "[" in answer and "]" in answer
            if confidence_score < 0.35 or not has_citations:
                hallucination_risk = "medium" if confidence_score >= 0.35 else "high"
            else:
                hallucination_risk = "low"

    # Audit Trail Logging (Security & Compliance)
    record_audit_log(
        settings,
        query=question,
        search_mode=search_mode,
        client_ip=client_ip,
        hit_count=len(hits),
        confidence_score=confidence_score,
        hallucination_risk=hallucination_risk,
    )

    return {
        "answer": answer,
        "context": context,
        "llm_configured": llm.configured,
        "confidence_score": confidence_score,
        "hallucination_risk": hallucination_risk,
        "sources": [
            {
                "rank": index,
                "document_id": hit["document_id"],
                "source_name": hit["source_name"],
                "chunk_index": hit["chunk_index"],
                "score": hit.get("score") or hit.get("rrf_score", 0.0),
                "text": hit["text"],
                "metadata": hit["metadata"],
                "quality_score": hit["quality_score"],
            }
            for index, hit in enumerate(hits, start=1)
        ],
    }


def document_summary(settings: Settings, document_id: UUID) -> dict[str, Any] | None:
    row = get_document(settings, document_id)
    if row is None:
        return None
    return {
        "document_id": row["id"],
        "name": row["source_name"],
        "source_type": row["source_type"],
        "mime_type": row["mime_type"],
        "metadata": row["metadata"] or {},
        "quality": row["quality_report"],
        "chunk_count": row["chunk_count"],
    }


OFFICIAL_STYLE_PROMPTS = {
    "공문서_개조식": (
        "당신은 대한민국 행정기관 및 공공기관의 전문 공문서 작성관입니다.\n"
        "제공된 CONTEXT 자료만을 바탕으로, 공문서 표준 규격에 따라 공식 문서를 작성하십시오.\n"
        "[작성 원칙]\n"
        "1. 문체: '~합니다/했습니다' 등 서술형 종결어미를 절대 쓰지 말고, 개조식 명사형 종결어미('~추진함', '~보고함', '~협조 바람', '~배치함')를 엄격히 준수하십시오.\n"
        "2. 번호 체계: 대항목 '1.', 중항목 '가.', 소항목 '(1)', 세부항목 '(가)', 세세부항목 '1)' 순서로 계층 구조를 작성하십시오.\n"
        "3. 구조: 제목, 1. 추진 배경 및 목적, 2. 주요 내용 및 현황, 3. 향후 계획 및 조치사항 순으로 구성하십시오.\n"
        "4. 근거 표기: 활용한 근거 문맥의 번호를 [1], [2] 형태로 명기하십시오."
    ),
    "보고서_서술형": (
        "당신은 전문 정책 분석관입니다.\n"
        "제공된 CONTEXT 자료만을 바탕으로, 논리적이고 명확한 서술형 종합 분석 보고서를 작성하십시오.\n"
        "개요, 현황 분석, 문제점 및 시사점, 해결 방안 순으로 일목요연하게 작성하십시오."
    ),
    "요약표": (
        "당신은 데이터 정리 전문가입니다.\n"
        "제공된 CONTEXT 자료의 핵심 내용을 한눈에 파악할 수 있도록 마크다운 표(| 구분 | 내용 | 비고 |) 형식으로 일목요연하게 정리하십시오."
    ),
}


def draft_document(
    settings: Settings,
    *,
    title: str,
    instructions: str = "",
    style: str = "공문서_개조식",
    top_k: int = 5,
    target_document_ids: list[UUID] | None = None,
) -> dict[str, Any]:
    search_query = f"{title} {instructions}".strip()
    embedder = get_embedder(settings)
    query_embedding = embedder.embed_query(search_query)

    doc_id = target_document_ids[0] if (target_document_ids and len(target_document_ids) == 1) else None
    hits = search_chunks(settings, query_embedding, top_k=top_k, document_id=doc_id)
    context = build_context(hits)

    system_prompt = OFFICIAL_STYLE_PROMPTS.get(style, OFFICIAL_STYLE_PROMPTS["공문서_개조식"])
    user_prompt = f"문서 제목: {title}\n요구사항: {instructions or '기본 양식에 맞추어 작성할 것'}"

    llm = LLMClient(settings)
    content = llm.complete(user_prompt, context, system_prompt=system_prompt) if hits else "관련 근거 문서를 찾을 수 없습니다."

    return {
        "title": title,
        "style": style,
        "content": content or "문서 초안을 작성하지 못했습니다.",
        "sources": [
            {
                "rank": index,
                "document_id": hit["document_id"],
                "source_name": hit["source_name"],
                "chunk_index": hit["chunk_index"],
                "score": hit.get("score") or hit.get("rrf_score", 0.0),
                "text": hit["text"],
                "metadata": hit["metadata"],
                "quality_score": hit["quality_score"],
            }
            for index, hit in enumerate(hits, start=1)
        ],
    }


EXTRACTION_SCHEMAS: dict[str, dict[str, Any]] = {
    "공문서_메타데이터": {
        "description": "공문서에서 문서번호, 시행일자, 발신기관, 수신기관, 제목, 담당자, 핵심 요지 등을 구조화하여 추출",
        "sample": {
            "문서번호": "예: 행안부-2026-012호 (없을 경우 null)",
            "시행일자": "예: 2026-03-15",
            "발신기관": "예: 식품의약품안전처",
            "수신기관": "예: 전국 시·도지사 및 시장·군수·구청장",
            "문서제목": "문서의 공식 제목",
            "담당부서": "예: 식품안전정책과",
            "담당자": "예: 홍길동 사무관 (02-1234-5678)",
            "핵심요약": "문서의 핵심 행정 사항 3줄 이내 개조식 요약",
            "주요키워드": ["키워드1", "키워드2", "키워드3"],
        },
    },
    "행정처분_요약": {
        "description": "행정처분, 시정명령, 과태료, 영업정지 관련 문서에서 처분 대상과 위반 사실을 정밀 추출",
        "sample": {
            "대상자_또는_업체명": "예: (주)한국식품",
            "처분종류": "예: 영업정지 15일 / 과태료 300만원 / 시정명령",
            "위반법령": "예: 식품위생법 제44조제1항",
            "위반일자": "예: 2026-02-10",
            "처분일자": "예: 2026-03-01",
            "주요위반내용": "위반 사실에 대한 구체적 설명",
            "이행기한_또는_정지기간": "예: 2026-03-10 ~ 2026-03-24",
        },
    },
    "사업계획_요약": {
        "description": "사업계획서 또는 과업지시서에서 사업명, 예산, 기간, 주요 과업 및 목표 추출",
        "sample": {
            "사업명": "예: 2026년 공공데이터 품질관리 및 RAG AI 인프라 고도화",
            "총사업예산": "예: 1억 5,000만원",
            "사업기간": "예: 2026.04 ~ 2026.11 (8개월간)",
            "주관부서": "예: 디지털정보담당관",
            "추진목적": "사업을 추진하는 주요 배경 및 목적",
            "주요과업목록": ["과업1", "과업2", "과업3"],
            "기대효과": "사업 완료 후 기대되는 정량적/정성적 성과",
        },
    },
}


def extract_document_entities(
    settings: Settings,
    *,
    document_id: UUID | None = None,
    text: str | None = None,
    schema_type: str = "공문서_메타데이터",
) -> dict[str, Any]:
    """Extract structured JSON entities from a stored document or raw text."""
    import json
    import re

    source_name = "direct_input"
    target_text = ""

    if document_id is not None:
        doc = get_document(settings, document_id)
        if not doc:
            raise ValueError(f"ID가 {document_id}인 문서를 찾을 수 없습니다.")
        source_name = doc["source_name"]
        # Use full content or top chunks
        chunks = get_document_chunks(settings, document_id)
        if chunks:
            target_text = "\n\n".join(c["content"] for c in chunks[:10])
        else:
            target_text = doc.get("content", "")
    elif text:
        target_text = text.strip()
    else:
        raise ValueError("document_id 또는 text 중 하나는 반드시 입력해야 합니다.")

    if not target_text:
        raise ValueError("분석할 텍스트 내용이 비어 있습니다.")

    schema_info = EXTRACTION_SCHEMAS.get(schema_type, EXTRACTION_SCHEMAS["공문서_메타데이터"])
    schema_sample = json.dumps(schema_info["sample"], ensure_ascii=False, indent=2)

    system_prompt = (
        "당신은 공공기관 및 기업 문서 분석과 정형 데이터 추출을 전담하는 고성능 AI입니다.\n"
        "제공된 본문 텍스트에서 정보를 정밀하게 추출하여, 반드시 아래 스키마 형식의 유효한 JSON 객체 하나만 출력하십시오.\n"
        "절대로 마크다운 코드블록(```json 등)이나 인사말, 설명 등 부가 텍스트를 붙이지 말고 오직 순수 JSON 문자열만 출력하십시오.\n"
        f"[추출 스키마 규격]:\n{schema_sample}"
    )

    llm = LLMClient(settings)
    raw_response = llm.complete(
        f"다음 문서에서 '{schema_type}' 정보를 추출하십시오:\n\n{target_text[:5000]}",
        "",
        system_prompt=system_prompt,
    )

    extracted_dict: dict[str, Any]
    if raw_response:
        # Strip potential markdown fences
        cleaned = raw_response.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            extracted_dict = json.loads(cleaned)
        except Exception:
            # Fallback: find outer braces
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    extracted_dict = json.loads(match.group(0))
                except Exception:
                    extracted_dict = {"raw_output": raw_response, "parse_error": "JSON 파싱 실패"}
            else:
                extracted_dict = {"raw_output": raw_response, "parse_error": "JSON 구조 없음"}
    else:
        extracted_dict = {"error": "LLM 응답을 생성하지 못했습니다."}

    return {
        "source_name": source_name,
        "schema_type": schema_type,
        "extracted_data": extracted_dict,
    }


def list_all_documents(
    settings: Settings,
    *,
    limit: int = 50,
    offset: int = 0,
    search: str | None = None,
) -> dict[str, Any]:
    rows, total = list_documents(settings, limit=limit, offset=offset, search=search)
    items = []
    for r in rows:
        items.append(
            {
                "id": r["id"],
                "source_name": r["source_name"],
                "source_type": r["source_type"],
                "mime_type": r["mime_type"],
                "metadata": r.get("metadata") or {},
                "quality_report": r.get("quality_report") or {},
                "chunk_count": r.get("chunk_count", 0),
                "created_at": r["created_at"].isoformat() if hasattr(r.get("created_at"), "isoformat") else str(r.get("created_at", "")),
            }
        )
    return {
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


def get_chunks_for_document(settings: Settings, document_id: UUID) -> list[dict[str, Any]]:
    rows = get_document_chunks(settings, document_id)
    return [
        {
            "id": r["id"],
            "chunk_index": r["chunk_index"],
            "content": r["content"],
            "metadata": r.get("metadata") or {},
        }
        for r in rows
    ]


def remove_document(settings: Settings, document_id: UUID) -> bool:
    return delete_document(settings, document_id)


def get_stats(settings: Settings) -> dict[str, Any]:
    stats = get_system_stats(settings)
    llm = LLMClient(settings)
    stats["llm_configured"] = llm.configured
    stats["llm_model"] = settings.llm_model
    stats["llm_base_url"] = settings.llm_base_url
    return stats


def enterprise_quality_service(
    settings: Settings,
    *,
    text: str | None = None,
    document_id: UUID | None = None,
    auto_mask_pii: bool = True,
    apply_local_term_replacements: bool = False,
) -> dict[str, Any]:
    """Perform comprehensive 5-pillar data quality & PII evaluation."""
    target_text = ""
    source_name = "inline"
    if document_id is not None:
        doc = get_document(settings, document_id)
        if not doc:
            raise ValueError(f"ID가 {document_id}인 문서를 찾을 수 없습니다.")
        source_name = doc["source_name"]
        chunks = get_document_chunks(settings, document_id)
        if chunks:
            target_text = "\n\n".join(c["content"] for c in chunks)
        else:
            target_text = doc.get("content", "")
    elif text:
        target_text = text
    else:
        raise ValueError("text 또는 document_id 중 하나는 필요합니다.")

    return evaluate_enterprise_quality(
        target_text,
        source_name=source_name,
        auto_mask_pii=auto_mask_pii,
        apply_local_term_replacements=apply_local_term_replacements,
    )


def structured_quality_service(
    *,
    csv_text: str | None = None,
    records: list[dict[str, Any]] | None = None,
    primary_key_col: str | None = None,
) -> dict[str, Any]:
    """Diagnose structured tabular data (CSV / JSON records)."""
    return evaluate_structured_dataset(records=records, csv_text=csv_text, primary_key_col=primary_key_col)


def get_recent_audit_logs(settings: Settings, limit: int = 50) -> list[dict[str, Any]]:
    """Retrieve audit logs for compliance tracking."""
    return get_audit_logs(settings, limit=limit)

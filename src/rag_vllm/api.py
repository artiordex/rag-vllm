# =============================================================================
# 파일명: api.py
# 경로: src/rag_vllm/api.py
# 목적: vLLM 기반 문서·검색·품질진단·초안 API와 보안 경계 관리함
# 작성자: AI전략팀
# 작성일: 2026-09-30
# 수정일: 2026-09-30
# =============================================================================

"""vLLM 기반 문서·검색·품질진단·초안 API와 보안 경계 관리함"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import mimetypes
import os
import time
from contextlib import asynccontextmanager, nullcontext
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from pydantic import ValidationError

from .config import ChatClientCredential, get_settings
from .db import DatabaseError, app_source_lock, check_db, close_db_pool, init_db
from .embeddings import EmbeddingError
from .llm import LLMError, close_shared_llm_client
from .metrics import record_http_request, render_prometheus_metrics
from .models import (
    AuditLogItem,
    ChatDeleteRequest,
    ChatHealthResponse,
    ChatIngestQualitySummary,
    ChatIngestResponse,
    ChatIngestRequest,
    ChatQueryResponse,
    ChatQueryRequest,
    ChatStructuredQueryRequest,
    ChatStructuredQueryResponse,
    ChunkItem,
    DocumentListItem,
    DocumentListResponse,
    DocumentResponse,
    DraftRequest,
    DraftResponse,
    EnterpriseQualityRequest,
    EnterpriseQualityResponse,
    ExtractionRequest,
    ExtractionResponse,
    GuardrailValidateInputRequest,
    GuardrailValidateOutputRequest,
    GuardrailValidateResponse,
    IngestResponse,
    LmopsFeedbackRequest,
    LmopsFeedbackResponse,
    LmopsMetricsSummaryResponse,
    LmopsTraceItem,
    QualityDiagnoseRequest,
    QualityReport,
    QueryOptions,
    QueryRequest,
    QueryResponse,
    RagEvalRequest,
    RagEvalResponse,
    StructuredQualityRequest,
    StructuredQualityResponse,
    StructuredQueryRequest,
    StructuredQueryResponse,
    TextIngestRequest,
    VectorStoreStatusResponse,
)
from .eval.deepeval_adapter import evaluate_with_deepeval
from .eval.local_evaluator import LocalRagEvaluator
from .guardrails import GuardrailAction, InputGuardrail, OutputGuardrail
from .lmops import get_lmops
from .vector_stores import get_vector_store
from .moe_dq_client import (
    MAX_MOE_UPLOAD_BYTES,
    MoeDqUpstreamError,
    compare_unstructured_uploads,
    diagnose_unstructured_upload,
    get_c2_term_matches,
)
from .parsers import MAX_DOCUMENT_INPUT_BYTES, ParseError, parse_document
from .quality import diagnose_text
from .standardization import detect_and_mask_pii
from .service import (
    diagnose_document_quality,
    document_summary,
    draft_document,
    enterprise_quality_service,
    extract_document_entities,
    get_chunks_for_document,
    get_recent_audit_logs,
    get_stats,
    ingest_text,
    list_all_documents,
    query_rag,
    query_structured_data,
    remove_document,
    remove_scoped_document,
    stream_query_rag,
    structured_quality_service,
)

logger = logging.getLogger(__name__)
settings = get_settings()


def _require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    """설정된 API 키가 있을 때 요청 헤더를 검증함

    Raises:
        HTTPException: 키가 없거나 설정값과 다를 때 발생함
    """
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="X-API-Key가 필요합니다.")


def _require_chat_client(
    x_rag_client_id: str | None = Header(default=None, alias="X-RAG-Client-ID"),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> ChatClientCredential:
    """앱 전용 토큰만 검증하고 서버 설정의 고정 검색 범위를 반환함"""
    if not settings.chat_clients:
        raise HTTPException(status_code=503, detail="앱별 RAG 챗봇 연동이 설정되지 않았습니다.")
    if (
        not x_rag_client_id
        or len(x_rag_client_id) > 64
        or not x_api_key
        or len(x_api_key) > 1024
    ):
        raise HTTPException(status_code=401, detail="RAG 앱 인증 헤더가 필요합니다.")

    credential = next((item for item in settings.chat_clients if item.client_id == x_rag_client_id), None)
    if credential is None:
        raise HTTPException(status_code=401, detail="RAG 앱 인증에 실패했습니다.")
    presented_hash = hashlib.sha256(x_api_key.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(presented_hash, credential.token_sha256):
        raise HTTPException(status_code=401, detail="RAG 앱 인증에 실패했습니다.")
    return credential


def _service_error(exc: Exception) -> HTTPException:
    """내부 예외를 외부에 노출 가능한 HTTP 오류로 변환함

    Returns:
        HTTPException: 오류 종류에 대응하는 상태 코드와 안전한 메시지임

    Caveats:
        매핑되지 않은 예외는 상세 내용을 숨기고 서버 로그에만 기록함
    """
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, ParseError):
        return HTTPException(status_code=415, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, (DatabaseError, EmbeddingError, LLMError, ConnectionError)):
        return HTTPException(status_code=503, detail=str(exc))
    logger.exception("rag-vllm request failed")
    return HTTPException(status_code=500, detail="요청을 처리하지 못했습니다.")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """애플리케이션 시작 시 PostgreSQL 스키마를 초기화하고 종료 시 커넥션 풀을 해제함

    Caveats:
        DB 자동 초기화 실패는 API 프로세스를 중단하지 않고 상태 확인에서
        degraded로 표시하도록 위임함
    """
    if settings.auto_init_db and settings.vector_store_type != "cpp_engine":
        try:
            init_db(settings)
        except DatabaseError as exc:
            logger.warning("DB 자동 초기화를 건너뜁니다: %s", exc)
    try:
        yield
    finally:
        if settings.vector_store_type == "cpp_engine":
            try:
                get_vector_store(settings, "cpp_engine").close()
            except Exception:
                logger.debug("C++ 벡터 저장소 종료 정리 중 오류", exc_info=True)
        close_db_pool(settings)
        close_shared_llm_client()


from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="rag-vllm API",
    version="0.1.0",
    description="Unstructured document ingestion, quality diagnostics, and pgvector RAG retrieval.",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

# SECURITY: 브라우저 교차 출처 접근은 기본 차단하고 신뢰된 origin만 허용함
cors_origins = [
    origin.strip()
    for origin in settings.cors_origins.split(",")
    if origin.strip()
]
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "X-API-Key"],
    )


@app.middleware("http")
async def metrics_middleware(request: Request, call_next: Any) -> Response:
    """모든 HTTP 요청의 상태 코드와 처리 시간을 계측하여 메트릭 레지스트리에 기록함"""
    start_time = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start_time
    record_http_request(request.method, request.url.path, response.status_code, duration)
    return response


from .dashboard import get_dashboard_html

router = APIRouter(dependencies=[Depends(_require_api_key)])
chat_router = APIRouter()


def _chat_source_name(client_id: str, name: str) -> str:
    """앱 이름으로 namespace를 분리하고 출처 이름의 개인정보 후보를 마스킹함"""
    if name != name.strip() or any(ord(character) < 32 for character in name):
        raise HTTPException(status_code=422, detail="문서 이름은 앞뒤 공백이나 제어 문자를 포함할 수 없습니다.")
    source_name = f"app://{client_id}/{name}"
    if len(source_name) > 512:
        raise HTTPException(status_code=422, detail="문서 이름이 허용 길이를 초과했습니다.")
    safe_source_name, _ = detect_and_mask_pii(source_name, mask=True)
    return safe_source_name


def _mask_app_chat_response(result: dict[str, Any], project_name: str) -> dict[str, Any]:
    """앱 응답에 필요한 필드만 남기고 답변·출처를 다시 마스킹·제한함"""
    masked_any = False
    answer = result.get("answer")
    if isinstance(answer, str):
        safe_answer, findings = detect_and_mask_pii(answer, mask=True)
        result["answer"] = safe_answer[:4_000]
        masked_any = masked_any or bool(findings)

    raw_sources = result.get("sources")
    if isinstance(raw_sources, list):
        safe_sources: list[dict[str, Any]] = []
        for raw_source in raw_sources:
            if not isinstance(raw_source, dict):
                raise HTTPException(
                    status_code=502,
                    detail="RAG 검색 근거 응답 형식을 확인할 수 없습니다.",
                )
            source_metadata = raw_source.get("metadata", {})
            if (
                not isinstance(source_metadata, dict)
                or (
                    "project_name" in source_metadata
                    and source_metadata.get("project_name") != project_name
                )
            ):
                # PGVector scopes by the parent document row, so chunk metadata
                # can omit project_name. When a backend returns it, reject any
                # contradiction instead of relabeling a cross-project hit.
                raise HTTPException(
                    status_code=502,
                    detail="RAG 검색 근거의 앱 범위를 확인할 수 없습니다.",
                )
            source_name = raw_source.get("source_name")
            text = raw_source.get("text")
            if not isinstance(source_name, str) or not isinstance(text, str):
                raise HTTPException(
                    status_code=502,
                    detail="RAG 검색 근거 응답 형식을 확인할 수 없습니다.",
                )
            raw_score = raw_source.get("score")
            if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
                raise HTTPException(
                    status_code=502,
                    detail="RAG 검색 근거의 점수를 확인할 수 없습니다.",
                )
            try:
                score = float(raw_score)
            except (OverflowError, ValueError) as exc:
                raise HTTPException(
                    status_code=502,
                    detail="RAG 검색 근거의 점수를 확인할 수 없습니다.",
                ) from exc
            if not math.isfinite(score):
                raise HTTPException(
                    status_code=502,
                    detail="RAG 검색 근거의 점수를 확인할 수 없습니다.",
                )
            safe_name, name_findings = detect_and_mask_pii(source_name, mask=True)
            safe_text, text_findings = detect_and_mask_pii(text, mask=True)
            masked_any = masked_any or bool(name_findings) or bool(text_findings)
            safe_sources.append(
                {
                    "rank": raw_source.get("rank"),
                    "document_id": raw_source.get("document_id"),
                    "source_name": safe_name[:256],
                    "chunk_index": raw_source.get("chunk_index"),
                    # Third-party reranker logits are not probabilities. Expose
                    # a stable bounded transport score while preserving rank.
                    "score": min(1.0, max(0.0, score)),
                    "text": safe_text[:1_200],
                    "metadata": {"project_name": project_name},
                }
            )
        result["sources"] = safe_sources

    if masked_any and result.get("guardrail_action") in {None, "allow", "mask"}:
        result["guardrail_action"] = "mask"
        violations = result.get("guardrail_violations")
        if not isinstance(violations, list):
            violations = []
        violations.append({"category": "pii", "message": "응답의 개인정보 후보를 다시 마스킹했습니다."})
        result["guardrail_violations"] = violations
    # Evaluation objects contain the full query and answer, while `context` repeats
    # complete retrieved chunks. Neither is needed by the app-facing chatbot API.
    return {
        "answer": result.get("answer"),
        "sources": result.get("sources", []),
        "confidence_score": result.get("confidence_score"),
        "hallucination_risk": result.get("hallucination_risk"),
        "trace_id": result.get("trace_id"),
        "guardrail_action": result.get("guardrail_action"),
    }


@chat_router.post("/chat/ingest", response_model=ChatIngestResponse)
def chat_ingest_endpoint(
    request: ChatIngestRequest,
    credential: ChatClientCredential = Depends(_require_chat_client),
) -> ChatIngestResponse:
    """앱 전용 자격증명으로 승인 파생 텍스트를 고정 앱 코퍼스에 등록함"""
    if "ingest" not in credential.permissions:
        raise HTTPException(status_code=403, detail="이 RAG 앱에는 문서 색인 권한이 없습니다.")
    safe_source_name = _chat_source_name(credential.client_id, request.name)
    try:
        ingest_settings = replace(
            settings,
            ingest_auto_mask_pii=True,
            ingest_apply_local_term_replacements=False,
        )
        lock_context = (
            nullcontext()
            if settings.vector_store_type == "cpp_engine"
            else app_source_lock(settings, project_name=credential.project_name, source_name=safe_source_name)
        )
        with lock_context:
            result = ingest_text(
                ingest_settings,
                name=safe_source_name,
                text=request.text,
                source_type="approved-derived-text",
                mime_type="text/plain",
                metadata={
                    "project_name": credential.project_name,
                    "ingestion_channel": "app-chat-approved-derived",
                },
                replace_existing_source=request.replace_existing_source,
            )
        quality = result.get("quality")
        if not isinstance(quality, dict):
            raise RuntimeError("RAG 색인 품질 집계를 생성하지 못했습니다.")
        try:
            return ChatIngestResponse(
                document_id=result["document_id"],
                chunk_count=result["chunk_count"],
                quality=ChatIngestQualitySummary(
                    score=quality["score"],
                    grade=quality["grade"],
                    pii_detected_count=quality["pii_detected_count"],
                ),
            )
        except ValidationError as exc:
            logger.error("앱 색인 응답 계약이 유효하지 않습니다.")
            raise HTTPException(status_code=502, detail="RAG 색인 응답을 확인할 수 없습니다.") from exc
    except Exception as exc:
        raise _service_error(exc) from exc


@chat_router.post("/chat/delete")
def chat_delete_endpoint(
    request: ChatDeleteRequest,
    credential: ChatClientCredential = Depends(_require_chat_client),
) -> dict[str, bool]:
    """앱별 토큰의 고정 코퍼스 안에서 해당 출처 이름을 제거함"""
    if "delete" not in credential.permissions:
        raise HTTPException(status_code=403, detail="이 RAG 앱에는 문서 삭제 권한이 없습니다.")
    source_name = _chat_source_name(credential.client_id, request.name)
    try:
        lock_context = (
            nullcontext()
            if settings.vector_store_type == "cpp_engine"
            else app_source_lock(settings, project_name=credential.project_name, source_name=source_name)
        )
        with lock_context:
            deleted = remove_scoped_document(
                settings,
                source_name=source_name,
                project_name=credential.project_name,
            )
        return {"deleted": deleted}
    except Exception as exc:
        raise _service_error(exc) from exc


@chat_router.post("/chat/query", response_model=ChatQueryResponse)
def chat_query_endpoint(
    request: ChatQueryRequest,
    http_request: Request,
    credential: ChatClientCredential = Depends(_require_chat_client),
) -> ChatQueryResponse:
    """앱 자격 증명에 연결된 승인 코퍼스만 검색하는 서버 간 챗봇 API임"""
    if "query" not in credential.permissions:
        raise HTTPException(status_code=403, detail="이 RAG 앱에는 질의 권한이 없습니다.")
    try:
        # 일반 LMOps 설정이 바뀌어도 앱별 챗봇의 질문·답변 원문은 저장하지 않음.
        chat_settings = replace(
            settings,
            guardrails_enabled=True,
            guardrails_block_on_injection=True,
            guardrails_mask_pii=True,
            lmops_store_query_text=False,
            lmops_store_answer_text=False,
            lmops_store_user_id=False,
        )
        client_ip = http_request.client.host if http_request.client else None
        result = query_rag(
            chat_settings,
            question=request.question,
            options=QueryOptions(
                top_k=request.top_k,
                project_name=credential.project_name,
                search_mode="hybrid",
                use_llm=request.generate_answer,
                client_ip=client_ip,
                enable_guardrails=True,
                enable_semantic_cache=False,
            ),
        )
        result = _mask_app_chat_response(result, credential.project_name)
        if not request.generate_answer:
            result["hallucination_risk"] = None
        try:
            return ChatQueryResponse(**result)
        except ValidationError as exc:
            logger.error("앱 질의 응답 계약이 유효하지 않습니다.")
            raise HTTPException(status_code=502, detail="RAG 질의 응답을 확인할 수 없습니다.") from exc
    except Exception as exc:
        raise _service_error(exc) from exc


@chat_router.post("/chat/query/structured", response_model=ChatStructuredQueryResponse)
def chat_structured_query_endpoint(
    request: ChatStructuredQueryRequest,
    http_request: Request,
    credential: ChatClientCredential = Depends(_require_chat_client),
) -> ChatStructuredQueryResponse:
    """앱 토큰의 고정 project_name 범위에 있는 표만 DuckDB로 집계함"""
    if "query" not in credential.permissions:
        raise HTTPException(status_code=403, detail="이 RAG 앱에는 질의 권한이 없습니다.")
    try:
        client_ip = http_request.client.host if http_request.client else None
        chat_settings = replace(
            settings,
            guardrails_enabled=True,
            guardrails_block_on_injection=True,
            guardrails_mask_pii=True,
        )
        result = query_structured_data(
            chat_settings,
            question=request.question,
            document_id=request.document_id,
            table_index=request.table_index,
            project_name=credential.project_name,
            client_ip=client_ip,
        )
        return ChatStructuredQueryResponse(
            answer=result["answer"],
            document_id=request.document_id,
            project_name=credential.project_name,
            plan=result["plan"],
            columns=result["columns"],
            row_count=result["row_count"],
            results=result["results"],
        )
    except Exception as exc:
        raise _service_error(exc) from exc


@chat_router.get("/chat/health", response_model=ChatHealthResponse)
def chat_health_endpoint(
    credential: ChatClientCredential = Depends(_require_chat_client),
) -> ChatHealthResponse:
    """앱별 query 자격증명으로 RAG 저장소 준비 상태를 확인함"""
    if "query" not in credential.permissions:
        raise HTTPException(status_code=403, detail="이 RAG 앱에는 질의 권한이 없습니다.")
    if settings.vector_store_type == "cpp_engine":
        healthy = get_vector_store(settings, "cpp_engine").health_check()
        return ChatHealthResponse(status="ok" if healthy else "degraded", database="not_required")
    try:
        database_ok = check_db(settings)
    except DatabaseError:
        database_ok = False
    return ChatHealthResponse(status="ok" if database_ok else "degraded", database="up" if database_ok else "down")


def dashboard_html() -> HTMLResponse:
    """선택적으로 활성화한 내부 POC 대시보드 HTML을 반환함"""
    return HTMLResponse(content=get_dashboard_html(), status_code=200)


if settings.dashboard_enabled:
    app.add_api_route(
        "/",
        dashboard_html,
        methods=["GET", "HEAD"],
        response_class=HTMLResponse,
        include_in_schema=False,
    )


@app.get("/metrics", response_class=PlainTextResponse)
def metrics_endpoint() -> PlainTextResponse:
    """Prometheus 표준 스크랩용 서비스 지표 텍스트를 반환함"""
    return PlainTextResponse(content=render_prometheus_metrics(settings), media_type="text/plain; version=0.0.4")


@app.get("/health", dependencies=[Depends(_require_api_key)])
def health() -> dict[str, Any]:
    """DB와 임베딩·LLM 설정의 현재 상태를 반환함

    Returns:
        dict[str, Any]: 서비스 상태, DB 연결 상태, 모델 설정 요약임
    """
    if settings.vector_store_type == "cpp_engine":
        try:
            store = get_vector_store(settings, "cpp_engine")
            return {
                "status": "ok" if store.health_check() else "degraded",
                "database": "not_required",
                "vector_store": "cpp_engine",
                "embedding_provider": settings.embedding_provider,
                "embedding_model": settings.embedding_model,
                "llm_configured": bool(settings.llm_base_url and settings.llm_model),
            }
        except Exception as exc:
            return {"status": "degraded", "database": "not_required", "detail": str(exc)}
    try:
        database_ok = check_db(settings)
    except DatabaseError as exc:
        return {"status": "degraded", "database": "down", "detail": str(exc)}
    return {
        "status": "ok" if database_ok else "degraded",
        "database": "up" if database_ok else "down",
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "llm_configured": bool(settings.llm_base_url and settings.llm_model),
    }


@app.get("/stats", dependencies=[Depends(_require_api_key)])
def stats_endpoint() -> dict[str, Any]:
    """문서·청크 수와 임베딩 설정 통계를 반환함"""
    try:
        return get_stats(settings)
    except Exception as exc:
        raise _service_error(exc) from exc


@router.get("/documents", response_model=DocumentListResponse)
def list_documents_endpoint(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    search: str | None = Query(default=None, max_length=512),
) -> DocumentListResponse:
    """검색어와 페이지 조건을 적용한 문서 목록을 반환함

    Args:
        limit: 페이지 최대 문서 수임
        offset: 건너뛸 문서 수임
        search: 문서명·본문 검색어임

    Returns:
        DocumentListResponse: 현재 페이지와 전체 건수임
    """
    try:
        return list_all_documents(settings, limit=limit, offset=offset, search=search)
    except Exception as exc:
        raise _service_error(exc) from exc


@router.post("/documents/text", response_model=IngestResponse)
def ingest_text_endpoint(request: TextIngestRequest) -> IngestResponse:
    """텍스트 본문을 정규화·진단·청킹·임베딩 후 저장함"""
    try:
        return ingest_text(
            settings,
            name=request.name,
            text=request.text,
            source_type=request.source_type,
            mime_type=request.mime_type,
            metadata=request.metadata,
            replace_existing_source=request.replace_existing_source,
        )
    except Exception as exc:
        raise _service_error(exc) from exc


@router.post("/documents/file", response_model=IngestResponse)
async def ingest_file_endpoint(
    file: UploadFile = File(...),
    metadata: str = Form(default="{}"),
    replace_existing_source: bool = Form(default=False),
) -> IngestResponse:
    """업로드 파일을 크기 제한 안에서 파싱하고 RAG 저장소에 등록함

    Caveats:
        파서가 확인한 메타데이터가 호출자 값보다 우선하며, 업로드 스트림은
        요청 종료 시 닫힘
    """
    try:
        upload_limit = min(settings.max_upload_bytes, MAX_DOCUMENT_INPUT_BYTES)
        payload = await file.read(upload_limit + 1)
        if len(payload) > upload_limit:
            raise HTTPException(status_code=413, detail=f"파일은 {upload_limit}바이트 이하만 허용됩니다.")
        parsed_metadata = json.loads(metadata)
        if not isinstance(parsed_metadata, dict):
            raise ValueError("metadata는 JSON object여야 합니다.")
        parsed = parse_document(file.filename or "uploaded-file", payload, file.content_type)
        # NOTE: 파서가 확인한 파일 사실값이 호출자 입력보다 우선해야 메타데이터 위조를 줄일 수 있음
        merged_metadata = {**parsed_metadata, **parsed.metadata}
        return ingest_text(
            settings,
            name=file.filename or "uploaded-file",
            text=parsed.text,
            source_type="file",
            mime_type=parsed.mime_type,
            metadata=merged_metadata,
            replace_existing_source=replace_existing_source,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise _service_error(exc) from exc
    finally:
        await file.close()


@router.get("/documents/{document_id}", response_model=DocumentResponse)
def get_document_endpoint(document_id: UUID) -> DocumentResponse:
    """문서 요약과 품질 리포트를 ID로 조회함

    Raises:
        HTTPException: 문서가 없거나 저장소 조회에 실패할 때 발생함
    """
    try:
        result = document_summary(settings, document_id)
    except Exception as exc:
        raise _service_error(exc) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="문서를 찾지 못했습니다.")
    return result


@router.get("/documents/{document_id}/chunks", response_model=list[ChunkItem])
def get_document_chunks_endpoint(document_id: UUID) -> list[ChunkItem]:
    """문서의 원문 순서 청크 목록을 반환함"""
    try:
        return get_chunks_for_document(settings, document_id)
    except Exception as exc:
        raise _service_error(exc) from exc


@router.delete("/documents/{document_id}")
def delete_document_endpoint(document_id: UUID) -> dict[str, Any]:
    """문서와 연결된 청크를 삭제하고 삭제 여부를 반환함"""
    try:
        deleted = remove_document(settings, document_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="삭제할 문서를 찾지 못했습니다.")
        return {"deleted": True, "document_id": str(document_id)}
    except Exception as exc:
        raise _service_error(exc) from exc


@router.post("/documents/extract", response_model=ExtractionResponse)
def extract_endpoint(request: ExtractionRequest) -> ExtractionResponse:
    """저장 문서 또는 입력 텍스트에서 구조화 항목을 추출함"""
    if request.document_id is not None and request.text is not None:
        raise HTTPException(status_code=422, detail="document_id와 text 중 하나만 지정해야 합니다.")
    try:
        return extract_document_entities(
            settings,
            document_id=request.document_id,
            text=request.text,
            schema_type=request.schema_type,
        )
    except Exception as exc:
        raise _service_error(exc) from exc


@router.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest, http_request: Request) -> QueryResponse:
    """질의를 dense·sparse 검색하고 선택적으로 vLLM 답변까지 생성함

    Caveats:
        부서와 보안 등급은 검색 필터이며 호출자의 실제 권한을 대체하지 않음
    """
    try:
        client_ip = http_request.client.host if http_request.client else None
        return query_rag(settings, question=request.question, options=request.to_options(client_ip))
    except Exception as exc:
        raise _service_error(exc) from exc


@router.post("/query/structured", response_model=StructuredQueryResponse)
def structured_query_endpoint(request: StructuredQueryRequest, http_request: Request) -> StructuredQueryResponse:
    """CSV·레코드 또는 등록된 CSV/XLSX 표 문서를 DuckDB 집계로 분석함"""
    try:
        client_ip = http_request.client.host if http_request.client else None
        return query_structured_data(
            settings,
            question=request.question,
            document_id=request.document_id,
            csv_text=request.csv_text,
            csv_delimiter=request.csv_delimiter,
            records=request.records,
            table_index=request.table_index,
            project_name=request.project_name,
            department=request.department,
            max_security_level=request.max_security_level,
            client_ip=client_ip,
        )
    except Exception as exc:
        raise _service_error(exc) from exc


@router.post("/query/stream")
def query_stream_endpoint(request: QueryRequest, http_request: Request) -> StreamingResponse:
    """질의 검색 및 vLLM 추론 토큰을 Server-Sent Events(SSE)로 실시간 스트리밍함

    Caveats:
        스트리밍 응답 중 발생하는 오류는 SSE 이벤트(type: error) 형태로 전송됨
    """
    try:
        client_ip = http_request.client.host if http_request.client else None
        stream_generator = stream_query_rag(
            settings,
            question=request.question,
            options=request.to_options(client_ip),
        )

        def stream_with_error_events():
            final_guardrail_sent = False
            done_sent = False

            def encode_event(payload: dict[str, Any]) -> str:
                return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            def incomplete_guardrail_event() -> str:
                return encode_event({
                    "type": "guardrail",
                    "phase": "final",
                    "action": GuardrailAction.BLOCK.value,
                    "faithfulness": 0.0,
                    "risk_score": 1.0,
                    "risk_level": "high",
                    "citation_validity": {
                        "valid": False,
                        "valid_citations": [],
                        "invalid_citations": [],
                    },
                    "violations": [{"category": "stream_incomplete", "severity": "high"}],
                })

            try:
                for event in stream_generator:
                    if event.startswith("data:"):
                        data = event.partition("data:")[2].strip()
                        if data == "[DONE]":
                            if not final_guardrail_sent:
                                yield incomplete_guardrail_event()
                                final_guardrail_sent = True
                            done_sent = True
                        else:
                            try:
                                payload = json.loads(data)
                            except (TypeError, ValueError):
                                payload = None
                            if (
                                isinstance(payload, dict)
                                and payload.get("type") == "guardrail"
                                and payload.get("phase") == "final"
                            ):
                                final_guardrail_sent = True
                    yield event
            except Exception:
                logger.exception("rag-vllm streaming request failed")
                yield encode_event({"type": "error", "message": "질의를 처리하지 못했습니다."})
                if not final_guardrail_sent:
                    yield incomplete_guardrail_event()
                    final_guardrail_sent = True

            if not final_guardrail_sent:
                yield incomplete_guardrail_event()
            if not done_sent:
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_with_error_events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception as exc:
        raise _service_error(exc) from exc


@router.post("/quality/diagnose", response_model=QualityReport)
def quality_diagnose_endpoint(request: QualityDiagnoseRequest) -> QualityReport:
    """인라인 텍스트 또는 저장 문서의 기초 품질 baseline을 계산함"""
    if request.document_id is not None and request.text is not None:
        raise HTTPException(status_code=422, detail="document_id와 text 중 하나만 지정해야 합니다.")
    if request.document_id is not None:
        try:
            result = diagnose_document_quality(settings, request.document_id)
        except Exception as exc:
            raise _service_error(exc) from exc
        if result is None:
            raise HTTPException(status_code=404, detail="문서를 찾지 못했습니다.")
        return result
    if request.text is None:
        raise HTTPException(status_code=422, detail="document_id 또는 text 중 하나는 필요합니다.")
    return diagnose_text(request.text, request.name, request.metadata)


@router.post("/quality/moe/unstructured")
async def moe_unstructured_diagnose_endpoint(file: UploadFile = File(...)) -> dict[str, Any]:
    """업로드를 MOE 비영속 개발 진단 API로 전달하고 로컬 결과와 비교함

    Caveats:
        MOE 결과를 rag-vllm DB에 저장하지 않으며 두 시스템의 점수는 측정 범위가
        달라 합산·상호 대체할 수 없음
    """
    base_url = settings.moe_dq_api_base_url
    body = bytearray()
    upload_limit = min(settings.max_upload_bytes, MAX_MOE_UPLOAD_BYTES)
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if len(suffix) > 16 or (suffix and not suffix[1:].isalnum()):
        suffix = ""
    try:
        if not base_url:
            raise HTTPException(
                status_code=503,
                detail="MOE_DQ_API_BASE_URL을 설정하고 MOE 개발 API를 실행해야 합니다.",
            )
        while chunk := await file.read(64 * 1024):
            body.extend(chunk)
            if len(body) > upload_limit:
                raise HTTPException(
                    status_code=413,
                    detail=f"업로드 크기가 rag-vllm/MOE 연동 한도({upload_limit} bytes)를 초과했습니다.",
                )
        if not body:
            raise HTTPException(status_code=422, detail="빈 파일은 진단할 수 없습니다.")

        safe_filename = f"upload{suffix}"
        content_type = mimetypes.guess_type(safe_filename)[0] or "application/octet-stream"
        payload = bytes(body)

        local_comparison: dict[str, Any]
        try:
            parsed = parse_document(safe_filename, payload, content_type)
        except (ParseError, ValueError):
            local_comparison = {
                "available": False,
                "assessment_scope": "rag-vllm-local-text-heuristic-not-official",
                "reason": "rag-vllm에서 이 파일의 텍스트를 추출하지 못했습니다.",
                "stored_by_rag_vllm": False,
            }
        else:
            local_report = diagnose_text(parsed.text, safe_filename, parsed.metadata)
            local_comparison = {
                "available": True,
                "assessment_scope": local_report["assessment_scope"],
                "rule_set_version": local_report["rule_set_version"],
                "stored_by_rag_vllm": False,
                "report": local_report,
            }

        moe_result = await diagnose_unstructured_upload(
            base_url=base_url,
            timeout_seconds=settings.moe_dq_api_timeout_seconds,
            filename=safe_filename,
            content_type=content_type,
            content=payload,
        )
        moe_result["local_comparison"] = {
            "rag_vllm": local_comparison,
            "scores_comparable": False,
            "note": "두 진단은 측정 범위가 다르므로 점수를 합산하거나 상호 대체하지 마십시오.",
        }
        return moe_result
    except MoeDqUpstreamError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    finally:
        await file.close()


@router.post("/quality/moe/unstructured/compare")
async def moe_unstructured_compare_endpoint(
    original_file: UploadFile = File(...),
    cleaned_file: UploadFile = File(...),
) -> dict[str, Any]:
    """원본·정제 파일을 MOE 개발 비교 API에 임시 전달하고 비영속 결과를 반환함"""
    base_url = settings.moe_dq_api_base_url
    upload_limit = min(settings.max_upload_bytes, MAX_MOE_UPLOAD_BYTES)
    uploads = (original_file, cleaned_file)
    try:
        if not base_url:
            raise HTTPException(
                status_code=503,
                detail="MOE_DQ_API_BASE_URL을 설정하고 MOE 개발 API를 실행해야 합니다.",
            )

        contents: list[bytes] = []
        extensions: list[str] = []
        content_types: list[str] = []
        for upload in uploads:
            suffix = Path(upload.filename or "").suffix.lower()
            if len(suffix) > 16 or not suffix or not suffix[1:].isalnum():
                raise HTTPException(status_code=415, detail="지원되는 파일 확장자를 확인해 주세요.")
            payload = bytearray()
            while chunk := await upload.read(64 * 1024):
                payload.extend(chunk)
                if len(payload) > upload_limit:
                    raise HTTPException(
                        status_code=413,
                        detail=f"각 파일의 크기는 rag-vllm/MOE 연동 한도({upload_limit} bytes) 이하여야 합니다.",
                    )
            if not payload:
                raise HTTPException(status_code=422, detail="원본과 정제본은 빈 파일일 수 없습니다.")
            extensions.append(suffix)
            content_types.append(mimetypes.guess_type(f"upload{suffix}")[0] or "application/octet-stream")
            contents.append(bytes(payload))

        return await compare_unstructured_uploads(
            base_url=base_url,
            timeout_seconds=settings.moe_dq_api_timeout_seconds,
            original_extension=extensions[0],
            original_content_type=content_types[0],
            original_content=contents[0],
            cleaned_extension=extensions[1],
            cleaned_content_type=content_types[1],
            cleaned_content=contents[1],
        )
    except MoeDqUpstreamError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    finally:
        for upload in uploads:
            await upload.close()


@router.get("/quality/moe/c2-term-matches")
async def moe_c2_term_matches_endpoint(
    table: str = Query(min_length=1, max_length=128),
) -> dict[str, Any]:
    """기존 C-2 테이블의 용어 후보를 MOE에 읽기 전용으로 조회함

    Args:
        table: 조회할 C-2 테이블 이름임

    Returns:
        dict[str, Any]: 도메인 검토가 필요한 용어 후보임
    """
    base_url = settings.moe_dq_api_base_url
    if not base_url:
        raise HTTPException(
            status_code=503,
            detail="MOE_DQ_API_BASE_URL을 설정하고 MOE 개발 API를 실행해야 합니다.",
        )
    try:
        return await get_c2_term_matches(
            base_url=base_url,
            timeout_seconds=settings.moe_dq_api_timeout_seconds,
            table=table,
        )
    except MoeDqUpstreamError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


@router.post("/quality/enterprise", response_model=EnterpriseQualityResponse)
def enterprise_quality_endpoint(request: EnterpriseQualityRequest) -> EnterpriseQualityResponse:
    """개인정보 후보 마스킹과 로컬 용어 표준화 품질을 계산함"""
    if request.document_id is not None and request.text is not None:
        raise HTTPException(status_code=422, detail="document_id와 text 중 하나만 지정해야 합니다.")
    try:
        return enterprise_quality_service(
            settings,
            text=request.text,
            document_id=request.document_id,
            auto_mask_pii=request.auto_mask_pii,
            apply_local_term_replacements=request.apply_local_term_replacements,
        )
    except Exception as exc:
        raise _service_error(exc) from exc


@router.post("/quality/structured", response_model=StructuredQualityResponse)
def structured_quality_endpoint(request: StructuredQualityRequest) -> StructuredQualityResponse:
    """CSV 또는 레코드 목록의 구조 품질을 계산함"""
    if request.csv_text is not None and request.records is not None:
        raise HTTPException(status_code=422, detail="csv_text와 records 중 하나만 지정해야 합니다.")
    try:
        return structured_quality_service(
            csv_text=request.csv_text,
            records=request.records,
            primary_key_col=request.primary_key_col,
        )
    except Exception as exc:
        raise _service_error(exc) from exc


@router.get("/audit/logs", response_model=list[AuditLogItem])
def audit_logs_endpoint(limit: int = Query(default=50, ge=1, le=200)) -> list[AuditLogItem]:
    """개인정보를 비공개 처리한 최근 감사 로그를 반환함"""
    try:
        return get_recent_audit_logs(settings, limit=limit)
    except Exception as exc:
        raise _service_error(exc) from exc


@router.post("/draft", response_model=DraftResponse)
def draft_endpoint(request: DraftRequest) -> DraftResponse:
    """검색 근거를 사용해 지정 형식의 공식 문서 초안을 생성함"""
    try:
        return draft_document(
            settings,
            title=request.title,
            instructions=request.instructions,
            style=request.style,
            top_k=request.top_k,
            target_document_ids=request.target_document_ids,
            project_name=request.project_name,
        )
    except Exception as exc:
        raise _service_error(exc) from exc


# =============================================================================
# 가드레일, 정량 평가(Eval), LMOps, 다중 벡터 저장소 엔드포인트 정의함
# =============================================================================

@router.post("/guardrails/validate-input", response_model=GuardrailValidateResponse)
def validate_input_endpoint(request: GuardrailValidateInputRequest) -> GuardrailValidateResponse:
    """사용자 질의 또는 입력 텍스트의 탈옥·인젝션·개인정보 위반 여부를 검증함"""
    try:
        guard = InputGuardrail(
            block_on_injection=request.block_on_injection,
            mask_pii=request.mask_pii,
        )
        res = guard.validate(request.text)
        return GuardrailValidateResponse(
            passed=res.passed,
            action=res.action.value,
            risk_score=res.risk_score,
            violations=[asdict(v) for v in res.violations],
            sanitized_text=res.sanitized_text,
            metadata=res.metadata,
        )
    except Exception as exc:
        raise _service_error(exc) from exc


@router.post("/guardrails/validate-output", response_model=GuardrailValidateResponse)
def validate_output_endpoint(request: GuardrailValidateOutputRequest) -> GuardrailValidateResponse:
    """생성된 답변의 환각 위험, 개인정보 누출 및 유해성을 검증함"""
    try:
        guard = OutputGuardrail(mask_pii=request.mask_pii)
        res = guard.validate(request.answer, request.context_chunks)
        return GuardrailValidateResponse(
            passed=res.passed,
            action=res.action.value,
            risk_score=res.risk_score,
            violations=[asdict(v) for v in res.violations],
            sanitized_text=res.sanitized_text,
            metadata=res.metadata,
        )
    except Exception as exc:
        raise _service_error(exc) from exc


@router.post("/eval/rag", response_model=RagEvalResponse)
def eval_rag_endpoint(request: RagEvalRequest) -> RagEvalResponse:
    """단일 질의응답 샘플에 대해 오프라인 정량 평가 지표를 산출함"""
    try:
        if request.engine == "deepeval":
            deepeval_metrics = evaluate_with_deepeval(
                input_text=request.query,
                actual_output=request.answer,
                retrieval_context=request.contexts,
                expected_output=request.ground_truth,
                base_url=settings.eval_judge_base_url,
                model_name=settings.eval_judge_model,
            )
            faithfulness = deepeval_metrics.get("faithfulness", {}).get("score")
            answer_rel = deepeval_metrics.get("answer_relevancy", {}).get("score")
            if faithfulness is None or answer_rel is None:
                failed = [
                    name
                    for name, score in (
                        ("faithfulness", faithfulness),
                        ("answer_relevancy", answer_rel),
                    )
                    if score is None
                ]
                raise HTTPException(
                    status_code=503,
                    detail=f"DeepEval 지표 계산에 실패했습니다: {', '.join(failed)}",
                )
            local_details = LocalRagEvaluator().evaluate_sample(
                query=request.query,
                answer=request.answer,
                contexts=request.contexts,
                ground_truth=request.ground_truth,
            )
            overall = round((faithfulness + answer_rel) * 50.0, 1)
            return RagEvalResponse(
                query=request.query,
                answer=request.answer,
                faithfulness=float(faithfulness),
                answer_relevance=float(answer_rel),
                context_precision=local_details.context_precision,
                context_recall=local_details.context_recall,
                hallucination_risk=(
                    "low" if faithfulness >= 0.75 else "medium" if faithfulness >= 0.45 else "high"
                ),
                overall_score=overall,
                details=deepeval_metrics,
            )
        else:
            evaluator = LocalRagEvaluator()
            res = evaluator.evaluate_sample(
                query=request.query,
                answer=request.answer,
                contexts=request.contexts,
                ground_truth=request.ground_truth,
            )
            return RagEvalResponse(
                query=res.query,
                answer=res.answer,
                faithfulness=res.faithfulness,
                answer_relevance=res.answer_relevance,
                context_precision=res.context_precision,
                context_recall=res.context_recall,
                hallucination_risk=res.hallucination_risk,
                overall_score=res.overall_score,
                details=res.to_dict(),
            )
    except Exception as exc:
        raise _service_error(exc) from exc


@router.get("/lmops/traces", response_model=list[LmopsTraceItem])
def get_lmops_traces_endpoint(
    limit: int = Query(default=50, ge=1, le=200),
    action: str | None = Query(default=None),
) -> list[LmopsTraceItem]:
    """최근 LMOps 요청 트레이스 요약 목록을 반환함"""
    try:
        manager = get_lmops(settings)
        traces = manager.get_recent_traces(limit=limit, action_filter=action)
        return [
            LmopsTraceItem(
                id=str(t["id"]),
                client_ip=t.get("client_ip"),
                query_text=t.get("query_text", ""),
                answer_text=t.get("answer_text"),
                model_name=t.get("model_name"),
                vector_store_type=t.get("vector_store_type"),
                total_latency_ms=float(t.get("total_latency_ms", 0.0)),
                total_tokens=int(t.get("total_tokens", 0)),
                guardrail_action=t.get("guardrail_action"),
                hallucination_risk=t.get("hallucination_risk"),
                created_at=t.get("created_at"),
            )
            for t in traces
        ]
    except Exception as exc:
        raise _service_error(exc) from exc


@router.get("/lmops/metrics/summary", response_model=LmopsMetricsSummaryResponse)
def get_lmops_summary_endpoint() -> LmopsMetricsSummaryResponse:
    """LMOps 관측성 종합 지표(질의 수, 지연시간, 토큰 소비, 피드백 등)를 반환함"""
    try:
        manager = get_lmops(settings)
        data = manager.get_metrics_summary()
        return LmopsMetricsSummaryResponse(**data)
    except Exception as exc:
        raise _service_error(exc) from exc


@router.post("/lmops/feedback", response_model=LmopsFeedbackResponse)
def record_lmops_feedback_endpoint(request: LmopsFeedbackRequest) -> LmopsFeedbackResponse:
    """사용자가 제공한 질의응답 피드백을 기록함"""
    try:
        manager = get_lmops(settings)
        fb_id = manager.record_feedback(
            trace_id=request.trace_id,
            thumbs=request.thumbs,
            rating=request.rating,
            comment=request.comment,
            corrected_answer=request.corrected_answer,
            user_id=request.user_id,
        )
        return LmopsFeedbackResponse(feedback_id=fb_id, status="recorded")
    except Exception as exc:
        raise _service_error(exc) from exc


@router.post("/lmops/export-sft")
def export_lmops_sft_endpoint(
    output_path: str = "data/sft/feedback_curated_sft.json",
    min_rating: int = 4,
) -> dict[str, Any]:
    """긍정 피드백 또는 사용자가 수정한 고품질 질의응답을 SFT 파인튜닝용 데이터셋으로 내보냄"""
    try:
        manager = get_lmops(settings)
        return manager.export_feedback_to_sft(output_path=output_path, min_rating=min_rating)
    except Exception as exc:
        raise _service_error(exc) from exc


@router.get("/vector-stores/status", response_model=VectorStoreStatusResponse)
def get_vector_stores_status_endpoint() -> VectorStoreStatusResponse:
    """현재 활성화된 벡터 저장소 엔진 상태 및 통계를 확인함"""
    try:
        store = get_vector_store(settings)
        health = store.health_check()
        stats = store.get_stats()
        return VectorStoreStatusResponse(
            active_engine=settings.vector_store_type,
            health=health,
            stats=stats,
        )
    except Exception as exc:
        raise _service_error(exc) from exc


app.include_router(router)
app.include_router(chat_router)


def run() -> None:
    """환경 설정의 주소와 포트로 uvicorn 서버를 실행함"""
    import uvicorn

    uvicorn.run(
        "rag_vllm.api:app",
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", "11020")),
        reload=False,
    )

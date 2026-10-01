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

import json
import logging
import mimetypes
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse

from .config import get_settings
from .db import DatabaseError, check_db, close_db_pool, init_db
from .embeddings import EmbeddingError
from .llm import LLMError, close_shared_llm_client
from .metrics import record_http_request, render_prometheus_metrics
from .models import (
    AuditLogItem,
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
    QueryRequest,
    QueryResponse,
    RagEvalRequest,
    RagEvalResponse,
    StructuredQualityRequest,
    StructuredQualityResponse,
    TextIngestRequest,
    VectorStoreStatusResponse,
)
from dataclasses import asdict
from .eval.deepeval_adapter import evaluate_with_deepeval
from .eval.local_evaluator import LocalRagEvaluator
from .guardrails import GuardrailAction, InputGuardrail, OutputGuardrail
from .lmops import get_lmops
from .vector_stores import get_vector_store
from .moe_dq_client import (
    MAX_MOE_UPLOAD_BYTES,
    MoeDqUpstreamError,
    diagnose_unstructured_upload,
    get_c2_term_matches,
)
from .parsers import MAX_DOCUMENT_INPUT_BYTES, ParseError, parse_document
from .quality import diagnose_text
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
    remove_document,
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
    if settings.auto_init_db:
        try:
            init_db(settings)
        except DatabaseError as exc:
            logger.warning("DB 자동 초기화를 건너뜁니다: %s", exc)
    try:
        yield
    finally:
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


@app.get("/", response_class=HTMLResponse)
@app.head("/")
def dashboard_html() -> HTMLResponse:
    """내부 POC 대시보드 HTML을 인라인으로 반환함"""
    return HTMLResponse(content=get_dashboard_html(), status_code=200)


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
            try:
                yield from stream_generator
            except Exception:
                logger.exception("rag-vllm streaming request failed")
                yield f"data: {json.dumps({'type': 'error', 'message': '질의를 처리하지 못했습니다.'}, ensure_ascii=False)}\n\n"
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


def run() -> None:
    """환경 설정의 주소와 포트로 uvicorn 서버를 실행함"""
    import uvicorn

    uvicorn.run(
        "rag_vllm.api:app",
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", "11020")),
        reload=False,
    )

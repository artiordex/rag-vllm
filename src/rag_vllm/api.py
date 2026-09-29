"""FastAPI application for ingestion, retrieval, and quality diagnostics."""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile

from .config import get_settings
from .db import DatabaseError, check_db, init_db
from .embeddings import EmbeddingError
from .llm import LLMError
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
    IngestResponse,
    QualityDiagnoseRequest,
    QualityReport,
    QueryRequest,
    QueryResponse,
    StructuredQualityRequest,
    StructuredQualityResponse,
    TextIngestRequest,
)
from .parsers import ParseError, parse_document
from .quality import diagnose_text
from .service import (
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
    structured_quality_service,
)

logger = logging.getLogger(__name__)
settings = get_settings()


def _require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="X-API-Key가 필요합니다.")


def _service_error(exc: Exception) -> HTTPException:
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, ParseError):
        return HTTPException(status_code=415, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, (DatabaseError, EmbeddingError, LLMError)):
        return HTTPException(status_code=503, detail=str(exc))
    logger.exception("rag-vllm request failed")
    return HTTPException(status_code=500, detail="요청을 처리하지 못했습니다.")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if settings.auto_init_db:
        try:
            init_db(settings)
        except DatabaseError as exc:
            logger.warning("DB 자동 초기화를 건너뜁니다: %s", exc)
    yield


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

# Browser access is disabled by default. Add only explicit trusted origins when
# a separate frontend needs to call this API.
cors_origins = [
    origin.strip()
    for origin in os.getenv("RAG_CORS_ORIGINS", "").split(",")
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

from fastapi.responses import HTMLResponse
from .dashboard import get_dashboard_html

router = APIRouter(dependencies=[Depends(_require_api_key)])


@app.get("/", response_class=HTMLResponse)
@app.head("/")
def dashboard_html() -> HTMLResponse:
    return HTMLResponse(content=get_dashboard_html(), status_code=200)


@app.get("/health")
def health() -> dict[str, Any]:
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


@app.get("/stats")
def stats_endpoint() -> dict[str, Any]:
    try:
        return get_stats(settings)
    except Exception as exc:
        raise _service_error(exc) from exc


@router.get("/documents", response_model=DocumentListResponse)
def list_documents_endpoint(
    limit: int = 50,
    offset: int = 0,
    search: str | None = None,
) -> DocumentListResponse:
    try:
        return list_all_documents(settings, limit=limit, offset=offset, search=search)
    except Exception as exc:
        raise _service_error(exc) from exc


@router.post("/documents/text", response_model=IngestResponse)
def ingest_text_endpoint(request: TextIngestRequest) -> IngestResponse:
    try:
        return ingest_text(
            settings,
            name=request.name,
            text=request.text,
            source_type=request.source_type,
            mime_type=request.mime_type,
            metadata=request.metadata,
        )
    except Exception as exc:
        raise _service_error(exc) from exc


@router.post("/documents/file", response_model=IngestResponse)
async def ingest_file_endpoint(
    file: UploadFile = File(...),
    metadata: str = Form(default="{}"),
) -> IngestResponse:
    try:
        payload = await file.read(settings.max_upload_bytes + 1)
        if len(payload) > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail=f"파일은 {settings.max_upload_bytes}바이트 이하만 허용됩니다.")
        parsed_metadata = json.loads(metadata)
        if not isinstance(parsed_metadata, dict):
            raise ValueError("metadata는 JSON object여야 합니다.")
        parsed = parse_document(file.filename or "uploaded-file", payload, file.content_type)
        # Parser evidence wins over caller-supplied values with the same key.
        merged_metadata = {**parsed_metadata, **parsed.metadata}
        return ingest_text(
            settings,
            name=file.filename or "uploaded-file",
            text=parsed.text,
            source_type="file",
            mime_type=parsed.mime_type,
            metadata=merged_metadata,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise _service_error(exc) from exc
    finally:
        await file.close()


@router.get("/documents/{document_id}", response_model=DocumentResponse)
def get_document_endpoint(document_id: UUID) -> DocumentResponse:
    try:
        result = document_summary(settings, document_id)
    except Exception as exc:
        raise _service_error(exc) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="문서를 찾지 못했습니다.")
    return result


@router.get("/documents/{document_id}/chunks", response_model=list[ChunkItem])
def get_document_chunks_endpoint(document_id: UUID) -> list[ChunkItem]:
    try:
        return get_chunks_for_document(settings, document_id)
    except Exception as exc:
        raise _service_error(exc) from exc


@router.delete("/documents/{document_id}")
def delete_document_endpoint(document_id: UUID) -> dict[str, Any]:
    try:
        deleted = remove_document(settings, document_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="삭제할 문서를 찾지 못했습니다.")
        return {"deleted": True, "document_id": str(document_id)}
    except Exception as exc:
        raise _service_error(exc) from exc


@router.post("/documents/extract", response_model=ExtractionResponse)
def extract_endpoint(request: ExtractionRequest) -> ExtractionResponse:
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
    try:
        client_ip = http_request.client.host if http_request.client else None
        return query_rag(
            settings,
            question=request.question,
            top_k=request.top_k,
            document_id=request.document_id,
            min_quality_score=request.min_quality_score,
            use_llm=request.use_llm,
            search_mode=request.search_mode,
            department=request.department,
            max_security_level=request.max_security_level,
            client_ip=client_ip,
        )
    except Exception as exc:
        raise _service_error(exc) from exc


@router.post("/quality/diagnose", response_model=QualityReport)
def quality_diagnose_endpoint(request: QualityDiagnoseRequest) -> QualityReport:
    if request.document_id is not None:
        try:
            result = document_summary(settings, request.document_id)
        except Exception as exc:
            raise _service_error(exc) from exc
        if result is None:
            raise HTTPException(status_code=404, detail="문서를 찾지 못했습니다.")
        return result["quality"]
    if request.text is None:
        raise HTTPException(status_code=422, detail="document_id 또는 text 중 하나는 필요합니다.")
    return diagnose_text(request.text, request.name, request.metadata)


@router.post("/quality/enterprise", response_model=EnterpriseQualityResponse)
def enterprise_quality_endpoint(request: EnterpriseQualityRequest) -> EnterpriseQualityResponse:
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
    try:
        return structured_quality_service(
            csv_text=request.csv_text,
            records=request.records,
            primary_key_col=request.primary_key_col,
        )
    except Exception as exc:
        raise _service_error(exc) from exc


@router.get("/audit/logs", response_model=list[AuditLogItem])
def audit_logs_endpoint(limit: int = 50) -> list[AuditLogItem]:
    try:
        return get_recent_audit_logs(settings, limit=limit)
    except Exception as exc:
        raise _service_error(exc) from exc


@router.post("/draft", response_model=DraftResponse)
def draft_endpoint(request: DraftRequest) -> DraftResponse:
    try:
        return draft_document(
            settings,
            title=request.title,
            instructions=request.instructions,
            style=request.style,
            top_k=request.top_k,
            target_document_ids=request.target_document_ids,
        )
    except Exception as exc:
        raise _service_error(exc) from exc


app.include_router(router)


def run() -> None:
    import uvicorn

    uvicorn.run(
        "rag_vllm.api:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "11020")),
        reload=False,
    )

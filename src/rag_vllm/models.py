"""API request and response models."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class TextIngestRequest(BaseModel):
    name: str = Field(min_length=1, max_length=512)
    text: str = Field(min_length=1)
    source_type: str = Field(default="text", max_length=64)
    mime_type: str | None = "text/plain"
    metadata: dict[str, Any] = Field(default_factory=dict)


class QualityIssue(BaseModel):
    severity: str
    code: str
    message: str
    value: float | int | str | None = None


class QualityReport(BaseModel):
    score: int
    status: str
    grade: str | None = None
    metrics: dict[str, float | int | str] = Field(default_factory=dict)
    issues: list[QualityIssue] = Field(default_factory=list)
    pillars: dict[str, Any] = Field(default_factory=dict)
    pii_detected_count: int = 0
    pii_details: list[dict[str, Any]] = Field(default_factory=list)
    standardized_terms_count: int = 0
    standardized_terms: dict[str, int] = Field(default_factory=dict)
    standardization_status: str | None = None
    standardization_source: str | None = None
    standardization_applied: bool | None = None
    pii_scan_scope: str | None = None
    pii_masking_applied: bool | None = None
    recommendations: list[str] = Field(default_factory=list)


class IngestResponse(BaseModel):
    document_id: UUID
    name: str
    chunk_count: int
    duplicate: bool
    quality: QualityReport


class SourceHit(BaseModel):
    rank: int
    document_id: UUID
    source_name: str
    chunk_index: int
    score: float
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    quality_score: int | None = None


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    document_id: UUID | None = None
    min_quality_score: int | None = Field(default=None, ge=0, le=100)
    department: str | None = Field(default=None, description="부서별 보안 접근 권한 필터 (예: '기획팀', '전사공통')")
    max_security_level: int | None = Field(default=None, ge=1, le=5, description="사용자 인가 보안 등급 (1: 일반 ~ 5: 극비)")
    use_llm: bool = True
    search_mode: str = Field(
        default="hybrid",
        description="검색 모드: 'hybrid' (하이브리드 RRF), 'dense' (벡터 유사도), 'sparse' (키워드)",
    )


class QueryResponse(BaseModel):
    answer: str | None
    context: str
    llm_configured: bool
    sources: list[SourceHit]
    confidence_score: float | None = Field(default=None, description="RAG 응답 신뢰도 점수 (0.0~1.0)")
    hallucination_risk: str | None = Field(default="low", description="환각 위험도: 'low', 'medium', 'high'")


class StructuredQualityRequest(BaseModel):
    csv_text: str | None = Field(default=None, description="CSV 형식 텍스트 데이터셋")
    records: list[dict[str, Any]] | None = Field(default=None, description="JSON 레코드 목록")
    primary_key_col: str | None = Field(default=None, description="유일성 검증 대상 기본키 컬럼명")


class StructuredQualityResponse(BaseModel):
    score: int
    grade: str
    total_rows: int
    total_columns: int
    columns: list[str]
    overall_null_ratio: float
    column_metrics: dict[str, Any]
    issues: list[str]


class AuditLogItem(BaseModel):
    id: int
    query: str
    search_mode: str | None
    client_ip: str | None
    hit_count: int
    confidence_score: float | None
    hallucination_risk: str | None
    created_at: str


class EnterpriseQualityRequest(BaseModel):
    text: str | None = None
    document_id: UUID | None = None
    auto_mask_pii: bool = True
    apply_local_term_replacements: bool = False


class EnterpriseQualityResponse(BaseModel):
    score: int
    grade: str
    status: str
    pillars: dict[str, int]
    pii_detected_count: int
    pii_details: list[dict[str, Any]] = Field(default_factory=list)
    standardized_terms_count: int
    standardized_terms: dict[str, int] = Field(default_factory=dict)
    standardization_status: str = "local_candidates_pending_review"
    standardization_source: str = "rag-vllm_builtin_synonyms"
    standardization_applied: bool = False
    pii_scan_scope: str = "heuristic_pattern_match"
    pii_masking_applied: bool = False
    recommendations: list[str] = Field(default_factory=list)


class QualityDiagnoseRequest(BaseModel):
    document_id: UUID | None = None
    name: str = Field(default="inline", max_length=512)
    text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentResponse(BaseModel):
    document_id: UUID
    name: str
    source_type: str
    mime_type: str | None
    metadata: dict[str, Any]
    quality: QualityReport
    chunk_count: int


class DocumentListItem(BaseModel):
    id: UUID
    source_name: str
    source_type: str
    mime_type: str | None
    metadata: dict[str, Any] = Field(default_factory=dict)
    quality_report: dict[str, Any] = Field(default_factory=dict)
    chunk_count: int
    created_at: Any | None = None


class DocumentListResponse(BaseModel):
    items: list[DocumentListItem]
    total: int
    limit: int
    offset: int


class ChunkItem(BaseModel):
    id: int
    chunk_index: int
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DraftRequest(BaseModel):
    title: str = Field(min_length=1, max_length=512, description="문서 제목 또는 작성할 보고서 주제")
    instructions: str = Field(default="", description="추가 작성 지침 또는 요구사항")
    style: str = Field(
        default="공문서_개조식",
        description="문체 스타일: '공문서_개조식', '보고서_서술형', '요약표'",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="RAG에서 검색할 참고 청크 수")
    target_document_ids: list[UUID] | None = Field(default=None, description="특정 문서만 참고할 경우 문서 ID 목록")


class DraftResponse(BaseModel):
    title: str
    style: str
    content: str
    sources: list[SourceHit]


class ExtractionRequest(BaseModel):
    document_id: UUID | None = None
    text: str | None = None
    schema_type: str = Field(
        default="공문서_메타데이터",
        description="추출 스키마: '공문서_메타데이터', '행정처분_요약', '사업계획_요약'",
    )


class ExtractionResponse(BaseModel):
    source_name: str
    schema_type: str
    extracted_data: dict[str, Any]

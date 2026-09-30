# =============================================================================
# 파일명: models.py
# 경로: src/rag_vllm/models.py
# 목적: 문서·검색·품질·감사·초안 API의 입력·출력 스키마 정의함
# 작성자: AI전략팀
# 작성일: 2026-09-30
# 수정일: 2026-09-30
# =============================================================================

"""문서·검색·품질·감사·초안 API의 입력·출력 스키마 정의함"""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from .chunking import MAX_TEXT_CHARACTERS
from .structured_quality import MAX_STRUCTURED_ROWS


class TextIngestRequest(BaseModel):
    """텍스트 문서 등록 요청 스키마임"""

    name: str = Field(min_length=1, max_length=512)
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARACTERS)
    source_type: str = Field(default="text", max_length=64)
    mime_type: str | None = "text/plain"
    metadata: dict[str, Any] = Field(default_factory=dict)


class QualityIssue(BaseModel):
    """품질 진단에서 발견한 단일 이슈 스키마임"""

    severity: str
    code: str
    message: str
    value: float | int | str | None = None


class QualityReport(BaseModel):
    """기초·기업형 품질 점수와 이슈·개인정보 결과를 담는 스키마임"""

    score: int
    status: str
    assessment_scope: str | None = None
    rule_set_version: str | None = None
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
    """문서 등록 결과와 중복·품질 요약을 담는 스키마임"""

    document_id: UUID
    name: str
    chunk_count: int
    duplicate: bool
    quality: QualityReport


class SourceHit(BaseModel):
    """검색된 단일 근거 청크의 순위와 품질 정보를 담는 스키마임"""

    rank: int
    document_id: UUID
    source_name: str
    chunk_index: int
    score: float
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    quality_score: int | None = None


class QueryRequest(BaseModel):
    """RAG 질문과 검색·보안 범위·생성 조건을 담는 요청 스키마임"""

    question: str = Field(min_length=1, max_length=5_000)
    top_k: int = Field(default=5, ge=1, le=50)
    document_id: UUID | None = None
    min_quality_score: int | None = Field(default=None, ge=0, le=100)
    department: str | None = Field(
        default=None,
        max_length=128,
        description="검색 부서 범위 필터(인증 사용자 권한 검증은 아님)",
    )
    max_security_level: int | None = Field(
        default=None,
        ge=1,
        le=5,
        description="검색할 보안 등급 상한(호출자가 지정하는 필터이며 접근통제는 아님)",
    )
    use_llm: bool = True
    search_mode: Literal["hybrid", "dense"] = Field(
        default="hybrid",
        description="검색 모드: 'hybrid' (하이브리드 RRF) 또는 'dense' (벡터 유사도)",
    )


class QueryResponse(BaseModel):
    """RAG 답변과 검색 문맥·근거·신뢰도 정보를 담는 응답 스키마임"""

    answer: str | None
    context: str
    llm_configured: bool
    sources: list[SourceHit]
    confidence_score: float | None = Field(default=None, description="RAG 응답 신뢰도 점수 (0.0~1.0)")
    hallucination_risk: str | None = Field(default="low", description="환각 위험도: 'low', 'medium', 'high'")


class StructuredQualityRequest(BaseModel):
    """CSV 또는 레코드 목록 구조 품질진단 요청 스키마임"""

    csv_text: str | None = Field(
        default=None,
        max_length=MAX_TEXT_CHARACTERS,
        description="최대 5,000,000자 CSV 데이터셋; 최대 100,000행·512컬럼·1,000,000셀",
    )
    records: list[dict[str, Any]] | None = Field(
        default=None,
        max_length=MAX_STRUCTURED_ROWS,
        description="최대 100,000행의 스칼라 JSON 레코드; 최대 512컬럼·1,000,000셀·5,000,000자",
    )
    primary_key_col: str | None = Field(default=None, max_length=512, description="유일성 검증 대상 기본키 컬럼명")


class StructuredQualityResponse(BaseModel):
    """구조 품질 점수와 컬럼별 진단 결과를 담는 응답 스키마임"""

    score: int
    grade: str
    assessment_scope: str = "rag-vllm-local-heuristic-not-official"
    rule_set_version: str = "structured-local-v3"
    total_rows: int
    total_columns: int
    columns: list[str]
    row_schema_mismatch_count: int = 0
    overall_null_ratio: float
    column_metrics: dict[str, Any]
    issues: list[str]


class AuditLogItem(BaseModel):
    """검색·보안 검토용 감사 로그 항목 스키마임"""

    id: int
    query: str = Field(description="원문 생략 표시와 문자 수 또는 레거시 비공개 표시")
    search_mode: str | None
    client_ip: str | None
    hit_count: int
    confidence_score: float | None
    hallucination_risk: str | None
    created_at: str


class EnterpriseQualityRequest(BaseModel):
    """개인정보 마스킹과 행정 용어 표준화 요청 스키마임"""

    text: str | None = Field(default=None, max_length=MAX_TEXT_CHARACTERS)
    document_id: UUID | None = None
    auto_mask_pii: bool = True
    apply_local_term_replacements: bool = False


class EnterpriseQualityResponse(BaseModel):
    """기업형 품질 점수·개인정보·표준화 결과를 담는 응답 스키마임"""

    score: int
    grade: str
    status: str
    assessment_scope: str = "rag-vllm-local-heuristic-not-official"
    rule_set_version: str = "enterprise-local-v1"
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
    """인라인 텍스트 또는 저장 문서 품질진단 요청 스키마임"""

    document_id: UUID | None = None
    name: str = Field(default="inline", max_length=512)
    text: str | None = Field(default=None, max_length=MAX_TEXT_CHARACTERS)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentResponse(BaseModel):
    """문서 상세 메타데이터와 품질·청크 요약을 담는 스키마임"""

    document_id: UUID
    name: str
    source_type: str
    mime_type: str | None
    metadata: dict[str, Any]
    quality: QualityReport
    chunk_count: int


class DocumentListItem(BaseModel):
    """문서 목록의 요약 항목 스키마임"""

    id: UUID
    source_name: str
    source_type: str
    mime_type: str | None
    metadata: dict[str, Any] = Field(default_factory=dict)
    quality_report: dict[str, Any] = Field(default_factory=dict)
    chunk_count: int
    created_at: Any | None = None


class DocumentListResponse(BaseModel):
    """페이지 단위 문서 목록과 전체 건수를 담는 스키마임"""

    items: list[DocumentListItem]
    total: int
    limit: int
    offset: int


class ChunkItem(BaseModel):
    """문서에 속한 검색 청크의 상세 스키마임"""

    id: int
    chunk_index: int
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DraftRequest(BaseModel):
    """검색 근거 기반 공식 문서 초안 요청 스키마임"""

    title: str = Field(min_length=1, max_length=512, description="문서 제목 또는 작성할 보고서 주제")
    instructions: str = Field(default="", max_length=5_000, description="추가 작성 지침 또는 요구사항")
    style: str = Field(
        default="공문서_개조식",
        description="문체 스타일: '공문서_개조식', '보고서_서술형', '요약표'",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="RAG에서 검색할 참고 청크 수")
    target_document_ids: list[UUID] | None = Field(
        default=None,
        max_length=50,
        description="특정 문서만 참고할 경우 최대 50개 문서 ID 목록",
    )


class DraftResponse(BaseModel):
    """공식 문서 초안과 사용 근거를 담는 응답 스키마임"""

    title: str
    style: str
    content: str
    sources: list[SourceHit]


class ExtractionRequest(BaseModel):
    """저장 문서 또는 원문에서 구조화 항목 추출을 요청하는 스키마임"""

    document_id: UUID | None = None
    text: str | None = Field(default=None, max_length=MAX_TEXT_CHARACTERS)
    schema_type: str = Field(
        default="공문서_메타데이터",
        description="추출 스키마: '공문서_메타데이터', '행정처분_요약', '사업계획_요약'",
    )


class ExtractionResponse(BaseModel):
    """구조화 추출 결과와 사용된 원문 범위를 담는 응답 스키마임"""

    source_name: str
    schema_type: str
    extracted_data: dict[str, Any]

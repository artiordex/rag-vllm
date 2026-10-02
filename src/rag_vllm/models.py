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

from pydantic import BaseModel, Field, model_validator

from .chunking import MAX_TEXT_CHARACTERS
from .structured_quality import MAX_STRUCTURED_ROWS


class TextIngestRequest(BaseModel):
    """텍스트 문서 등록 요청 스키마임"""

    name: str = Field(min_length=1, max_length=512)
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARACTERS)
    source_type: str = Field(default="text", max_length=64)
    mime_type: str | None = "text/plain"
    metadata: dict[str, Any] = Field(default_factory=dict)
    replace_existing_source: bool = Field(
        default=False,
        description="같은 source name의 이전 색인을 새 내용으로 교체할지 여부임",
    )


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


from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QueryOptions:
    """RAG 질의의 검색 필터와 캐시·가드레일 옵션을 캡슐화함"""

    top_k: int = 5
    document_id: UUID | None = None
    min_quality_score: int | None = None
    project_name: str | None = None
    department: str | None = None
    max_security_level: int | None = None
    search_mode: str = "hybrid"
    use_llm: bool = True
    client_ip: str | None = None
    enable_guardrails: bool | None = None
    vector_store_type: str | None = None
    enable_semantic_cache: bool = True
    lora_name: str | None = None
    query_mode: Literal["auto", "retrieval", "structured_sql"] = "auto"


class QueryRequest(BaseModel):
    """RAG 질문과 검색·보안 범위·생성 조건을 담는 요청 스키마임"""

    question: str = Field(min_length=1, max_length=5_000)
    top_k: int = Field(default=5, ge=1, le=50)
    document_id: UUID | None = None
    project_name: str | None = Field(
        default=None,
        max_length=128,
        description="프로젝트 문서만 검색할 때 사용하는 프로젝트 디렉터리 이름임",
    )
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
    query_mode: Literal["auto", "retrieval", "structured_sql"] = "auto"
    search_mode: Literal["hybrid", "dense"] = Field(
        default="hybrid",
        description="검색 모드: 'hybrid' (하이브리드 RRF) 또는 'dense' (벡터 유사도)",
    )
    enable_guardrails: bool | None = Field(default=None, description="가드레일 검사 활성화 여부임")
    vector_store_type: Literal["pgvector", "qdrant", "weaviate", "cpp_engine"] | None = Field(
        default=None,
        description="VECTOR_STORE_TYPE 설정과 동일해야 하는 벡터 저장소",
    )
    lora_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
        description="서버 허용 목록에 등록된 vLLM LoRA 어댑터 별칭",
    )

    def to_options(self, client_ip: str | None = None) -> QueryOptions:
        """QueryRequest 스키마를 내부 서비스용 QueryOptions 불변 객체로 변환함"""
        return QueryOptions(
            top_k=self.top_k,
            document_id=self.document_id,
            min_quality_score=self.min_quality_score,
            project_name=self.project_name,
            department=self.department,
            max_security_level=self.max_security_level,
            search_mode=self.search_mode,
            use_llm=self.use_llm,
            query_mode=self.query_mode,
            client_ip=client_ip,
            enable_guardrails=self.enable_guardrails,
            vector_store_type=self.vector_store_type,
            lora_name=self.lora_name,
        )


class ChatQueryRequest(BaseModel):
    """앱별 검색 범위를 서버 설정에 맡기고 필요하면 LLM 답변 생성을 생략함"""

    model_config = {"extra": "forbid"}

    question: str = Field(min_length=1, max_length=5_000)
    top_k: int = Field(default=5, ge=1, le=10)
    generate_answer: bool = Field(default=True, strict=True)


class ChatIngestRequest(BaseModel):
    """앱 코퍼스에 승인 파생 텍스트를 등록하는 요청 스키마임"""

    model_config = {"extra": "forbid"}

    name: str = Field(min_length=1, max_length=440)
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARACTERS)
    replace_existing_source: bool = False


class ChatDeleteRequest(BaseModel):
    """앱 코퍼스 안의 문서 한 건을 출처 이름으로 제거하는 요청임"""

    model_config = {"extra": "forbid"}

    name: str = Field(min_length=1, max_length=440)


class ChatIngestQualitySummary(BaseModel):
    """앱 색인 결과에 필요한 품질 집계만 반환함"""

    score: int = Field(ge=0, le=100)
    grade: str = Field(min_length=1, max_length=64)
    pii_detected_count: int = Field(ge=0)


class ChatIngestResponse(BaseModel):
    """원문 조각·품질 상세를 제외한 앱용 색인 응답임"""

    document_id: UUID
    chunk_count: int = Field(ge=0)
    quality: ChatIngestQualitySummary


class ChatHealthResponse(BaseModel):
    """앱 자격증명으로 확인하는 최소 RAG 상태 응답임"""

    status: Literal["ok", "degraded"]
    database: Literal["up", "down", "not_required"]


class ChatSourceMetadata(BaseModel):
    """앱용 인용에는 서버가 고정한 코퍼스 이름만 포함함"""

    model_config = {"extra": "forbid"}

    project_name: str = Field(min_length=1, max_length=128)


class ChatSourceHit(BaseModel):
    """앱용 RAG 응답에서 반환하는 개인정보 마스킹·길이 제한 근거임"""

    rank: int = Field(ge=1, le=10)
    document_id: UUID
    source_name: str = Field(min_length=1, max_length=256)
    chunk_index: int = Field(ge=0)
    score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    text: str = Field(min_length=1, max_length=1_200)
    metadata: ChatSourceMetadata


class ChatQueryResponse(BaseModel):
    """앱용 검색 답변 계약으로 전체 문맥과 평가 트레이스를 제외함"""

    answer: str | None = Field(max_length=4_000)
    sources: list[ChatSourceHit] = Field(max_length=10)
    confidence_score: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    hallucination_risk: Literal["low", "medium", "high"] | None = None
    trace_id: str | None = Field(default=None, max_length=128)
    guardrail_action: Literal["allow", "mask", "flag", "block"] | None = None
    query_route: Literal["retrieval", "conversation"] | None = None


class ChatStructuredQueryRequest(BaseModel):
    """앱의 고정 코퍼스 안에 등록된 표 문서를 질의함"""

    model_config = {"extra": "forbid"}

    question: str = Field(min_length=1, max_length=5_000)
    document_id: UUID
    table_index: int = Field(default=0, ge=0, le=100)


class ChatStructuredQueryResponse(BaseModel):
    """프로젝트 스코프 표 집계 결과를 반환함"""

    model_config = {"extra": "forbid"}

    answer: str = Field(min_length=1, max_length=4_000)
    query_route: Literal["structured_sql"] = "structured_sql"
    document_id: UUID
    project_name: str = Field(min_length=1, max_length=128)
    plan: dict[str, Any]
    columns: list[str]
    row_count: int = Field(ge=0)
    results: list[dict[str, Any]] = Field(max_length=100)


class QueryResponse(BaseModel):
    """RAG 답변과 검색 문맥·근거·신뢰도 정보를 담는 응답 스키마임"""

    answer: str | None
    context: str
    llm_configured: bool
    sources: list[SourceHit]
    confidence_score: float | None = Field(default=None, description="RAG 응답 신뢰도 점수 (0.0~1.0)")
    hallucination_risk: str | None = Field(default="low", description="환각 위험도: 'low', 'medium', 'high'")
    attribution_details: dict[str, Any] | None = Field(default=None, description="인용 번호 및 근거 충실도 세부 지표임")
    trace_id: str | None = Field(default=None, description="LMOps 요청 트레이스 ID임")
    guardrail_action: str | None = Field(default=None, description="가드레일 판정 결과: allow, mask, flag, block")
    guardrail_violations: list[dict[str, Any]] | None = Field(default=None, description="탐지된 가드레일 위반 목록임")
    evaluation: dict[str, Any] | None = Field(default=None, description="로컬 RAG 정량 평가 지표 결과임")
    query_route: Literal["retrieval", "conversation", "structured_sql"] | None = None
    query_plan: dict[str, Any] | None = None
    retrieval_gate: dict[str, Any] | None = None
    structured_result: dict[str, Any] | None = None


class StructuredQueryRequest(BaseModel):
    """지정 문서 또는 입력 표를 DuckDB 집계 경로로 질의함"""

    model_config = {"extra": "forbid"}

    question: str = Field(min_length=1, max_length=5_000)
    document_id: UUID | None = None
    csv_text: str | None = Field(default=None, max_length=MAX_TEXT_CHARACTERS)
    csv_delimiter: Literal[",", "\t"] = ","
    records: list[dict[str, Any]] | None = Field(default=None, max_length=MAX_STRUCTURED_ROWS)
    table_index: int = Field(default=0, ge=0, le=100)
    project_name: str | None = Field(default=None, max_length=128)
    department: str | None = Field(default=None, max_length=128)
    max_security_level: int | None = Field(default=None, ge=1, le=5)

    @model_validator(mode="after")
    def validate_single_data_source(self) -> "StructuredQueryRequest":
        supplied = sum(value is not None for value in (self.document_id, self.csv_text, self.records))
        if supplied != 1:
            raise ValueError("document_id, csv_text, records 중 정확히 하나를 지정해야 합니다.")
        return self


class StructuredQueryResponse(BaseModel):
    """DuckDB 집계 계획과 결과를 반환함"""

    answer: str
    query_route: Literal["structured_sql"] = "structured_sql"
    engine: Literal["duckdb"] = "duckdb"
    document_id: UUID | None = None
    source_name: str | None = None
    plan: dict[str, Any]
    sql: str
    columns: list[str]
    row_count: int = Field(ge=0)
    results: list[dict[str, Any]] = Field(max_length=100)


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
    project_name: str | None = Field(
        default=None,
        max_length=128,
        description="특정 프로젝트 문서만 참고할 때 사용하는 프로젝트 디렉터리 이름임",
    )
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
    guardrail_action: str = "allow"
    guardrail_violations: list[dict[str, str]] = Field(default_factory=list)


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
    guardrail_action: str = "allow"
    guardrail_violations: list[dict[str, str]] = Field(default_factory=list)


# =============================================================================
# 가드레일, 평가(Eval), LMOps, 벡터 저장소 연동 모델 정의함
# =============================================================================

class GuardrailValidateInputRequest(BaseModel):
    """입력 가드레일 검증 요청 스키마임"""

    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARACTERS)
    block_on_injection: bool = True
    mask_pii: bool = True


class GuardrailValidateOutputRequest(BaseModel):
    """출력 가드레일 검증 요청 스키마임"""

    answer: str = Field(min_length=1)
    context_chunks: list[dict[str, Any]] | None = None
    mask_pii: bool = True


class GuardrailValidateResponse(BaseModel):
    """가드레일 검증 응답 스키마임"""

    passed: bool
    action: str
    risk_score: float
    violations: list[dict[str, Any]]
    sanitized_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RagEvalRequest(BaseModel):
    """RAG 정량 평가 요청 스키마임"""

    query: str
    answer: str
    contexts: list[str]
    ground_truth: str | None = None
    engine: str = Field(default="local", description="평가 엔진: 'local' 또는 'deepeval'")


class RagEvalResponse(BaseModel):
    """RAG 정량 평가 응답 스키마임"""

    query: str
    answer: str
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float | None = None
    hallucination_risk: str
    overall_score: float
    details: dict[str, Any] = Field(default_factory=dict)


class LmopsFeedbackRequest(BaseModel):
    """사용자 질의응답 피드백 제출 스키마임"""

    trace_id: UUID
    thumbs: int | None = Field(default=None, description="1(추천) 또는 -1(비추천)")
    rating: int | None = Field(default=None, ge=1, le=5, description="1~5점 평점")
    comment: str | None = Field(default=None, max_length=1000)
    corrected_answer: str | None = Field(default=None, max_length=MAX_TEXT_CHARACTERS)
    user_id: str | None = None


class LmopsFeedbackResponse(BaseModel):
    """피드백 저장 응답 스키마임"""

    feedback_id: int
    status: str = "recorded"


class LmopsTraceItem(BaseModel):
    """LMOps 트레이스 요약 아이템 스키마임"""

    id: str
    client_ip: str | None
    query_text: str
    answer_text: str | None
    model_name: str | None
    vector_store_type: str | None
    total_latency_ms: float
    total_tokens: int
    guardrail_action: str | None
    hallucination_risk: str | None
    created_at: str | None


class LmopsMetricsSummaryResponse(BaseModel):
    """LMOps 관측성 종합 요약 통계 스키마임"""

    total_queries: int
    avg_latency_ms: float
    total_tokens: int
    estimated_cost_usd: float
    guardrail_block_rate: float
    high_hallucination_count: int
    feedback_count: int
    avg_rating: float
    thumbs_up: int
    thumbs_down: int


class VectorStoreStatusResponse(BaseModel):
    """벡터 저장소 가용성 및 통계 상태 스키마임"""

    active_engine: str
    health: bool
    stats: dict[str, Any]

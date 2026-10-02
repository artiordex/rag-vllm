# =============================================================================
# 파일명: config.py
# 경로: src/rag_vllm/config.py
# 목적: vLLM·RAG·MOE 연동 환경변수와 보안·입력 제약 검증함
# 작성자: AI전략팀
# 작성일: 2026-09-30
# 수정일: 2026-09-30
# =============================================================================

"""vLLM·RAG·MOE 연동 환경변수와 보안·입력 제약 검증함"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from urllib.parse import urlsplit

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # NOTE: 의존성 설치 전에도 설정 모듈을 불러올 수 있도록 선택 처리함
    pass


def _bool_env(name: str, default: bool) -> bool:
    """환경변수의 대표적인 참 값 표현을 bool로 변환함"""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _int_env(name: str, default: int) -> int:
    """환경변수의 정수값을 변환하고 잘못된 입력을 즉시 거부함"""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _float_env(name: str, default: float) -> float:
    """환경변수의 실수값을 변환하고 잘못된 입력을 즉시 거부함"""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


@dataclass(frozen=True, slots=True)
class ChatClientCredential:
    """앱별 RAG 검색 범위와 API 키 해시를 담는 서버 간 인증 항목임"""

    client_id: str
    project_name: str
    permissions: frozenset[str]
    token_sha256: str = field(repr=False)


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """중복 JSON 키를 거부해 앱 자격 설정의 모호한 해석을 막음"""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("RAG_CHAT_CLIENTS_JSON contains a duplicate key")
        result[key] = value
    return result


def _chat_clients_env() -> tuple[ChatClientCredential, ...]:
    """앱별 서버 간 토큰을 검증하고 평문 대신 SHA-256만 보관함"""
    raw = os.getenv("RAG_CHAT_CLIENTS_JSON", "").strip()
    if not raw:
        return ()

    try:
        parsed = json.loads(raw, object_pairs_hook=_reject_duplicate_json_keys)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("RAG_CHAT_CLIENTS_JSON must be valid JSON without duplicate keys") from exc
    if not isinstance(parsed, dict) or not parsed or len(parsed) > 100:
        raise ValueError("RAG_CHAT_CLIENTS_JSON must be a non-empty object with at most 100 clients")

    clients: list[ChatClientCredential] = []
    seen_hashes: set[str] = set()
    seen_projects: set[str] = set()
    for client_id, value in parsed.items():
        if not isinstance(client_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{1,63}", client_id):
            raise ValueError("RAG_CHAT_CLIENTS_JSON client IDs must use lowercase letters, digits, '_' or '-'")
        if not isinstance(value, dict) or not {"token", "project_name"}.issubset(value) or set(value) - {
            "token",
            "project_name",
            "permissions",
        }:
            raise ValueError("Each RAG chat client must contain token, project_name, and optional permissions")

        token = value["token"]
        project_name = value["project_name"]
        if (
            not isinstance(token, str)
            or len(token) < 32
            or len(token) > 1024
            or token.strip() != token
            or any(character.isspace() for character in token)
        ):
            raise ValueError("RAG chat client tokens must be 32 to 1024 non-whitespace characters")
        if (
            not isinstance(project_name, str)
            or not project_name
            or len(project_name) > 128
            or project_name.strip() != project_name
            or project_name == "*"
        ):
            raise ValueError("RAG chat client project_name must be a fixed non-empty project name")
        if project_name in seen_projects:
            raise ValueError("RAG chat client project_name scopes must be unique")

        raw_permissions = value.get("permissions", ["query"])
        if (
            not isinstance(raw_permissions, list)
            or not raw_permissions
            or any(not isinstance(permission, str) for permission in raw_permissions)
            or len(set(raw_permissions)) != len(raw_permissions)
            or not set(raw_permissions).issubset({"query", "ingest", "delete"})
            or "query" not in raw_permissions
        ):
            raise ValueError("RAG chat permissions must be a unique list containing query and optional ingest/delete")

        token_sha256 = hashlib.sha256(token.encode("utf-8")).hexdigest()
        if token_sha256 in seen_hashes:
            raise ValueError("RAG chat client tokens must be unique")
        seen_hashes.add(token_sha256)
        seen_projects.add(project_name)
        clients.append(ChatClientCredential(client_id, project_name, frozenset(raw_permissions), token_sha256))
    return tuple(clients)


def _llm_lora_adapters_env() -> tuple[tuple[str, str], ...]:
    """vLLM에 이미 등록된 LoRA 이름의 허용 목록을 읽음"""
    raw = os.getenv("LLM_LORA_ADAPTERS_JSON", "").strip()
    if not raw:
        return ()
    try:
        parsed = json.loads(raw, object_pairs_hook=_reject_duplicate_json_keys)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("LLM_LORA_ADAPTERS_JSON must be valid JSON without duplicate keys") from exc
    if not isinstance(parsed, dict) or not parsed or len(parsed) > 32:
        raise ValueError("LLM_LORA_ADAPTERS_JSON must be an object with at most 32 adapters")
    adapters: list[tuple[str, str]] = []
    for alias, model_name in parsed.items():
        if not isinstance(alias, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", alias):
            raise ValueError("LoRA aliases must contain only letters, digits, '.', '_' or '-'")
        if (
            not isinstance(model_name, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./:-]{0,127}", model_name)
        ):
            raise ValueError("LoRA model names must be non-empty vLLM model identifiers")
        adapters.append((alias, model_name))
    return tuple(adapters)


@dataclass(frozen=True, slots=True)
class Settings:
    """RAG 저장·임베딩·검색·vLLM·MOE 연동에 필요한 불변 설정 모음임"""

    database_url: str
    database_connect_timeout: int
    embedding_provider: str
    embedding_model: str
    embedding_dim: int
    embedding_device: str
    embedding_use_fp16: bool
    embedding_batch_size: int
    chunk_size: int
    chunk_overlap: int
    retrieval_top_k: int
    llm_base_url: str | None
    llm_api_key: str | None
    llm_model: str | None
    llm_timeout_seconds: float
    max_upload_bytes: int
    moe_dq_api_base_url: str | None
    moe_dq_api_timeout_seconds: float
    ingest_auto_mask_pii: bool
    ingest_apply_local_term_replacements: bool
    api_key: str | None
    auto_init_db: bool
    embedding_gpu_memory_fraction: float = 0.25
    embedding_max_concurrent_calls: int = 1
    parent_child_chunking_enabled: bool = False
    child_chunk_size: int = 280
    child_chunk_overlap: int = 40
    parent_chunk_size: int = 1200
    parent_chunk_overlap: int = 120
    hyde_enabled: bool = False
    query_router_enabled: bool = False
    cors_origins: str = ""
    database_min_pool_size: int = 1
    database_max_pool_size: int = 10
    database_pool_timeout: float = 30.0
    vector_store_type: str = "pgvector"
    qdrant_url: str = "http://127.0.0.1:6333"
    weaviate_url: str = "http://127.0.0.1:8080"
    weaviate_grpc_host: str = ""
    weaviate_grpc_port: int = 50051
    weaviate_api_key: str | None = None
    cpp_engine_library: str = ""
    cpp_engine_max_vectors: int = 25000
    guardrails_enabled: bool = True
    guardrails_block_on_injection: bool = True
    guardrails_mask_pii: bool = True
    lmops_enabled: bool = True
    lmops_log_file: str = "data/lmops_traces.jsonl"
    lmops_store_query_text: bool = False
    lmops_store_answer_text: bool = False
    lmops_store_feedback_text: bool = False
    lmops_store_user_id: bool = False
    eval_judge_model: str = "rag-vllm-model"
    eval_judge_base_url: str = "http://127.0.0.1:11435/v1"
    chat_clients: tuple[ChatClientCredential, ...] = ()
    contextual_retrieval_enabled: bool = False
    multi_query_enabled: bool = False
    crag_enabled: bool = False
    semantic_cache_redis_url: str | None = field(default=None, repr=False)
    semantic_cache_ttl_seconds: int = 1800
    semantic_cache_max_entries: int = 256
    structured_query_enabled: bool = True
    llm_lora_adapters: tuple[tuple[str, str], ...] = ()
    dashboard_enabled: bool = True

    def __post_init__(self) -> None:
        """인제스트나 검색을 사용할 수 없게 만드는 설정을 조기 검증함"""
        if self.database_connect_timeout <= 0:
            raise ValueError("DATABASE_CONNECT_TIMEOUT must be positive")
        if self.embedding_provider not in {"flag", "hash"}:
            raise ValueError("EMBEDDING_PROVIDER must be 'flag' or 'hash'")
        if self.embedding_dim <= 0:
            raise ValueError("EMBEDDING_DIM must be positive")
        if self.embedding_batch_size <= 0:
            raise ValueError("EMBEDDING_BATCH_SIZE must be positive")
        if not 0 < self.embedding_gpu_memory_fraction <= 1:
            raise ValueError("EMBEDDING_GPU_MEMORY_FRACTION must be in (0, 1]")
        if self.embedding_max_concurrent_calls <= 0:
            raise ValueError("EMBEDDING_MAX_CONCURRENT_CALLS must be positive")
        if self.child_chunk_size <= 0 or self.parent_chunk_size < self.child_chunk_size:
            raise ValueError("PARENT_CHUNK_SIZE must be >= CHILD_CHUNK_SIZE > 0")
        if not 0 <= self.child_chunk_overlap < self.child_chunk_size:
            raise ValueError("CHILD_CHUNK_OVERLAP must be between 0 and CHILD_CHUNK_SIZE - 1")
        if not 0 <= self.parent_chunk_overlap < self.parent_chunk_size:
            raise ValueError("PARENT_CHUNK_OVERLAP must be between 0 and PARENT_CHUNK_SIZE - 1")
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be positive")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be between 0 and CHUNK_SIZE - 1")
        if self.retrieval_top_k <= 0:
            raise ValueError("RETRIEVAL_TOP_K must be positive")
        if self.llm_timeout_seconds <= 0:
            raise ValueError("LLM_TIMEOUT_SECONDS must be positive")
        if self.max_upload_bytes <= 0:
            raise ValueError("MAX_UPLOAD_BYTES must be positive")
        if self.moe_dq_api_timeout_seconds <= 0:
            raise ValueError("MOE_DQ_API_TIMEOUT_SECONDS must be positive")
        if self.moe_dq_api_base_url:
            parsed = urlsplit(self.moe_dq_api_base_url)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(
                    "MOE_DQ_API_BASE_URL must be an http(s) base URL without credentials, query, or fragment"
                )
        if self.database_min_pool_size <= 0:
            raise ValueError("DATABASE_MIN_POOL_SIZE must be positive")
        if self.database_max_pool_size < self.database_min_pool_size:
            raise ValueError("DATABASE_MAX_POOL_SIZE must be >= DATABASE_MIN_POOL_SIZE")
        if self.database_pool_timeout <= 0:
            raise ValueError("DATABASE_POOL_TIMEOUT must be positive")
        weaviate = urlsplit(self.weaviate_url)
        if (
            weaviate.scheme not in {"http", "https"}
            or not weaviate.hostname
            or weaviate.username is not None
            or weaviate.password is not None
            or weaviate.path not in {"", "/"}
            or weaviate.query
            or weaviate.fragment
        ):
            raise ValueError("WEAVIATE_URL must be a host URL without credentials, path, query, or fragment")
        try:
            weaviate_port = weaviate.port
        except ValueError as exc:
            raise ValueError("WEAVIATE_URL must contain a valid TCP port") from exc
        if weaviate_port is not None and not 1 <= weaviate_port <= 65535:
            raise ValueError("WEAVIATE_URL port must be between 1 and 65535")
        if not 1 <= self.weaviate_grpc_port <= 65535:
            raise ValueError("WEAVIATE_GRPC_PORT must be between 1 and 65535")
        if self.weaviate_grpc_host and any(char in self.weaviate_grpc_host for char in "/?#@"):
            raise ValueError("WEAVIATE_GRPC_HOST must contain only a hostname")
        if self.cpp_engine_max_vectors <= 0 or self.cpp_engine_max_vectors > 2**32 - 1:
            raise ValueError("CPP_ENGINE_MAX_VECTORS must be between 1 and 2^32-1")
        if self.vector_store_type not in {"pgvector", "qdrant", "weaviate", "cpp_engine"}:
            raise ValueError("VECTOR_STORE_TYPE must be 'pgvector', 'qdrant', 'weaviate', or 'cpp_engine'")
        if self.semantic_cache_ttl_seconds <= 0:
            raise ValueError("SEMANTIC_CACHE_TTL_SECONDS must be positive")
        if not 1 <= self.semantic_cache_max_entries <= 4096:
            raise ValueError("SEMANTIC_CACHE_MAX_ENTRIES must be between 1 and 4096")
        if self.semantic_cache_redis_url:
            redis_url = urlsplit(self.semantic_cache_redis_url)
            if redis_url.scheme not in {"redis", "rediss"} or not redis_url.hostname or redis_url.fragment:
                raise ValueError("SEMANTIC_CACHE_REDIS_URL must be a redis(s) URL with a hostname")

    @classmethod
    def from_env(cls) -> "Settings":
        """환경변수와 기본값으로 유효한 설정 객체를 생성함

        Returns:
            Settings: 서비스 실행에 사용할 검증 완료 설정임

        Raises:
            ValueError: 숫자 형식·URL·서비스 제약을 만족하지 못할 때 발생함
        """
        return cls(
            database_url=os.getenv(
                "DATABASE_URL",
                "postgresql://ragvllm:ragvllm-dev-only@localhost:55433/ragvllm",
            ),
            database_connect_timeout=_int_env("DATABASE_CONNECT_TIMEOUT", 3),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "flag").strip().lower(),
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
            embedding_dim=_int_env("EMBEDDING_DIM", 1024),
            embedding_device=os.getenv("EMBEDDING_DEVICE", "cpu"),
            embedding_use_fp16=_bool_env("EMBEDDING_USE_FP16", False),
            embedding_batch_size=_int_env("EMBEDDING_BATCH_SIZE", 8),
            chunk_size=_int_env("CHUNK_SIZE", 1600),
            chunk_overlap=_int_env("CHUNK_OVERLAP", 240),
            retrieval_top_k=_int_env("RETRIEVAL_TOP_K", 5),
            llm_base_url=os.getenv("LLM_BASE_URL") or None,
            llm_api_key=os.getenv("LLM_API_KEY") or None,
            llm_model=os.getenv("LLM_MODEL") or None,
            llm_timeout_seconds=_float_env("LLM_TIMEOUT_SECONDS", 60.0),
            max_upload_bytes=_int_env("MAX_UPLOAD_BYTES", 20 * 1024 * 1024),
            moe_dq_api_base_url=(os.getenv("MOE_DQ_API_BASE_URL") or "").strip().rstrip("/") or None,
            moe_dq_api_timeout_seconds=_float_env("MOE_DQ_API_TIMEOUT_SECONDS", 30.0),
            ingest_auto_mask_pii=_bool_env("INGEST_AUTO_MASK_PII", True),
            ingest_apply_local_term_replacements=_bool_env("INGEST_APPLY_LOCAL_TERM_REPLACEMENTS", False),
            api_key=os.getenv("RAG_LAB_API_KEY") or None,
            auto_init_db=_bool_env("AUTO_INIT_DB", True),
            embedding_gpu_memory_fraction=_float_env("EMBEDDING_GPU_MEMORY_FRACTION", 0.25),
            embedding_max_concurrent_calls=_int_env("EMBEDDING_MAX_CONCURRENT_CALLS", 1),
            parent_child_chunking_enabled=_bool_env("PARENT_CHILD_CHUNKING_ENABLED", False),
            child_chunk_size=_int_env("CHILD_CHUNK_SIZE", 280),
            child_chunk_overlap=_int_env("CHILD_CHUNK_OVERLAP", 40),
            parent_chunk_size=_int_env("PARENT_CHUNK_SIZE", 1200),
            parent_chunk_overlap=_int_env("PARENT_CHUNK_OVERLAP", 120),
            hyde_enabled=_bool_env("HYDE_ENABLED", False),
            query_router_enabled=_bool_env("QUERY_ROUTER_ENABLED", False),
            cors_origins=os.getenv("RAG_CORS_ORIGINS", ""),
            database_min_pool_size=_int_env("DATABASE_MIN_POOL_SIZE", 1),
            database_max_pool_size=_int_env("DATABASE_MAX_POOL_SIZE", 10),
            database_pool_timeout=_float_env("DATABASE_POOL_TIMEOUT", 30.0),
            vector_store_type=os.getenv("VECTOR_STORE_TYPE", "pgvector").strip().lower(),
            qdrant_url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"),
            weaviate_url=os.getenv("WEAVIATE_URL", "http://127.0.0.1:8080"),
            weaviate_grpc_host=os.getenv("WEAVIATE_GRPC_HOST", "").strip(),
            weaviate_grpc_port=_int_env("WEAVIATE_GRPC_PORT", 50051),
            weaviate_api_key=os.getenv("WEAVIATE_API_KEY") or None,
            cpp_engine_library=os.getenv("CPP_VECTOR_ENGINE_LIBRARY", "").strip(),
            cpp_engine_max_vectors=_int_env("CPP_ENGINE_MAX_VECTORS", 25000),
            guardrails_enabled=_bool_env("GUARDRAILS_ENABLED", True),
            guardrails_block_on_injection=_bool_env("GUARDRAILS_BLOCK_ON_INJECTION", True),
            guardrails_mask_pii=_bool_env("GUARDRAILS_MASK_PII", True),
            lmops_enabled=_bool_env("LMOPS_ENABLED", True),
            lmops_log_file=os.getenv("LMOPS_LOG_FILE", "data/lmops_traces.jsonl"),
            lmops_store_query_text=_bool_env("LMOPS_STORE_QUERY_TEXT", False),
            lmops_store_answer_text=_bool_env("LMOPS_STORE_ANSWER_TEXT", False),
            lmops_store_feedback_text=_bool_env("LMOPS_STORE_FEEDBACK_TEXT", False),
            lmops_store_user_id=_bool_env("LMOPS_STORE_USER_ID", False),
            eval_judge_model=os.getenv("EVAL_JUDGE_MODEL", "rag-vllm-model"),
            eval_judge_base_url=os.getenv("EVAL_JUDGE_BASE_URL", "http://127.0.0.1:11435/v1"),
            chat_clients=_chat_clients_env(),
            contextual_retrieval_enabled=_bool_env("CONTEXTUAL_RETRIEVAL_ENABLED", False),
            multi_query_enabled=_bool_env("MULTI_QUERY_ENABLED", False),
            crag_enabled=_bool_env("CRAG_ENABLED", False),
            semantic_cache_redis_url=(os.getenv("SEMANTIC_CACHE_REDIS_URL") or "").strip() or None,
            semantic_cache_ttl_seconds=_int_env("SEMANTIC_CACHE_TTL_SECONDS", 1800),
            semantic_cache_max_entries=_int_env("SEMANTIC_CACHE_MAX_ENTRIES", 256),
            structured_query_enabled=_bool_env("STRUCTURED_QUERY_ENABLED", True),
            llm_lora_adapters=_llm_lora_adapters_env(),
            dashboard_enabled=_bool_env("RAG_DASHBOARD_ENABLED", True),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """프로세스 전체에서 재사용할 설정 객체를 반환함

    Caveats:
        환경변수 변경은 캐시를 지우거나 프로세스를 재시작해야 반영됨
    """
    return Settings.from_env()

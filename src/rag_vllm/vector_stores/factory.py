# =============================================================================
# 파일명: factory.py
# 경로: src/rag_vllm/vector_stores/factory.py
# 목적: 애플리케이션 설정에 따라 적합한 벡터 저장소 인스턴스를 생성하는 팩토리 제공함
# 작성자: AI전략팀
# 작성일: 2026-10-01
# 수정일: 2026-10-01
# =============================================================================

"""애플리케이션 설정에 따라 적합한 벡터 저장소 인스턴스를 생성하는 팩토리 제공함"""

from __future__ import annotations

import logging
import hashlib
import threading
from ..config import Settings
from .base import BaseVectorStore
from .cpp_engine_store import CppEngineVectorStore
from .pgvector_store import PgVectorStore
from .qdrant_store import QdrantVectorStore
from .weaviate_store import WeaviateVectorStore

logger = logging.getLogger(__name__)

_GLOBAL_STORES: dict[tuple[str, str, str, str, int, str, int, str, int, str], BaseVectorStore] = {}
_STORE_LOCK = threading.Lock()


def get_vector_store(settings: Settings, store_type_override: str | None = None) -> BaseVectorStore:
    """설정된 엔진 유형(pgvector, qdrant, weaviate)에 맞는 벡터 저장소 인스턴스를 반환함

    Args:
        settings: 애플리케이션 설정 객체임
        store_type_override: 명시적으로 특정 엔진을 지정할 때 사용함

    Returns:
        BaseVectorStore: 표준 벡터 저장소 어댑터 인스턴스임
    """
    engine = (store_type_override or getattr(settings, "vector_store_type", "pgvector")).strip().lower()
    cache_key = (
        engine,
        settings.database_url,
        settings.qdrant_url,
        settings.weaviate_url,
        int(settings.embedding_dim),
        settings.cpp_engine_library,
        int(settings.cpp_engine_max_vectors),
        settings.weaviate_grpc_host,
        int(settings.weaviate_grpc_port),
        hashlib.sha256((settings.weaviate_api_key or "").encode("utf-8")).hexdigest(),
    )
    store = _GLOBAL_STORES.get(cache_key)
    if store is not None and (engine != "cpp_engine" or store.health_check()):
        return store
    with _STORE_LOCK:
        store = _GLOBAL_STORES.get(cache_key)
        if store is None or (engine == "cpp_engine" and not store.health_check()):
            store = _create_store(engine, settings)
            _GLOBAL_STORES[cache_key] = store
    return store


def _create_store(engine: str, settings: Settings) -> BaseVectorStore:
    """엔진명에 따른 구현체 인스턴스를 생성함"""
    if engine == "qdrant":
        logger.info("Qdrant 벡터 저장소 활성화됨")
        store = QdrantVectorStore(settings)
        store.initialize()
        return store
    elif engine == "weaviate":
        logger.info("Weaviate 벡터 저장소 활성화됨")
        store = WeaviateVectorStore(settings, require_server=True)
        store.initialize()
        return store
    elif engine == "pgvector":
        logger.info("기본 pgvector 저장소 활성화됨")
        store = PgVectorStore(settings)
        return store
    elif engine == "cpp_engine":
        logger.info("사내 C++ 인메모리 벡터 저장소 활성화됨")
        store = CppEngineVectorStore(settings)
        store.initialize()
        return store
    raise ValueError(f"지원하지 않는 벡터 저장소 유형입니다: {engine}")

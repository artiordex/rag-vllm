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
import threading
from typing import Any

from ..config import Settings
from .base import BaseVectorStore
from .pgvector_store import PgVectorStore
from .qdrant_store import QdrantVectorStore
from .weaviate_store import WeaviateVectorStore

logger = logging.getLogger(__name__)

_GLOBAL_STORE: BaseVectorStore | None = None
_STORE_LOCK = threading.Lock()


def get_vector_store(settings: Settings, store_type_override: str | None = None) -> BaseVectorStore:
    """설정된 엔진 유형(pgvector, qdrant, weaviate)에 맞는 벡터 저장소 인스턴스를 반환함

    Args:
        settings: 애플리케이션 설정 객체임
        store_type_override: 명시적으로 특정 엔진을 지정할 때 사용함

    Returns:
        BaseVectorStore: 표준 벡터 저장소 어댑터 인스턴스임
    """
    global _GLOBAL_STORE
    engine = (store_type_override or getattr(settings, "vector_store_type", "pgvector")).strip().lower()

    if store_type_override:
        # 오버라이드 지정 시 새 인스턴스 반환함
        return _create_store(engine, settings)

    if _GLOBAL_STORE is None:
        with _STORE_LOCK:
            if _GLOBAL_STORE is None:
                _GLOBAL_STORE = _create_store(engine, settings)
    return _GLOBAL_STORE


def _create_store(engine: str, settings: Settings) -> BaseVectorStore:
    """엔진명에 따른 구현체 인스턴스를 생성함"""
    if engine == "qdrant":
        logger.info("Qdrant 벡터 저장소 활성화됨")
        store = QdrantVectorStore(settings)
        store.initialize()
        return store
    elif engine == "weaviate":
        logger.info("Weaviate 벡터 저장소 활성화됨")
        store = WeaviateVectorStore(settings)
        store.initialize()
        return store
    else:
        logger.info("기본 pgvector 저장소 활성화됨")
        store = PgVectorStore(settings)
        return store

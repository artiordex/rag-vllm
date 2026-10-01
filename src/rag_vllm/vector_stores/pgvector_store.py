# =============================================================================
# 파일명: pgvector_store.py
# 경로: src/rag_vllm/vector_stores/pgvector_store.py
# 목적: PostgreSQL pgvector 확장을 활용하는 벡터 저장소 어댑터 구현함
# 작성자: AI전략팀
# 작성일: 2026-10-01
# 수정일: 2026-10-01
# =============================================================================

"""PostgreSQL pgvector 확장을 활용하는 벡터 저장소 어댑터 구현함"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from ..config import Settings
from ..db import (
    check_db,
    delete_document as db_delete_document,
    get_system_stats,
    init_db,
    save_document as db_save_document,
    search_chunks as db_search_chunks,
    search_chunks_hybrid as db_search_chunks_hybrid,
)
from .base import BaseVectorStore

logger = logging.getLogger(__name__)


class PgVectorStore(BaseVectorStore):
    """PostgreSQL pgvector 기반의 벡터 저장소 구현체임"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def initialize(self) -> None:
        """데이터베이스 연결 및 pgvector 스키마를 초기화함"""
        init_db(self.settings)

    def health_check(self) -> bool:
        """PostgreSQL 헬스체크 쿼리를 실행함"""
        return check_db(self.settings)

    def save_document(
        self,
        *,
        source_name: str,
        source_type: str,
        mime_type: str | None,
        content_hash: str,
        content: str,
        metadata: dict[str, Any],
        quality_report: dict[str, Any],
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
        document_id: UUID | None = None,
        replace_existing_source: bool = False,
    ) -> tuple[UUID, bool, int]:
        """원천 문서 메타데이터 및 청크 벡터를 PostgreSQL에 저장함"""
        return db_save_document(
            self.settings,
            source_name=source_name,
            source_type=source_type,
            mime_type=mime_type,
            content_hash=content_hash,
            content=content,
            metadata=metadata,
            quality_report=quality_report,
            chunks=chunks,
            embeddings=embeddings,
            document_id=document_id,
            replace_existing_source=replace_existing_source,
        )

    def search(
        self,
        *,
        query_vector: list[float],
        top_k: int = 5,
        document_id: UUID | None = None,
        document_ids: list[UUID] | None = None,
        min_quality_score: int | None = None,
        project_name: str | None = None,
        department: str | None = None,
        max_security_level: int | None = None,
    ) -> list[dict[str, Any]]:
        """pgvector 코사인 거리 기반 밀집 검색을 수행함"""
        return db_search_chunks(
            self.settings,
            query_vector,
            top_k=top_k,
            document_id=document_id,
            document_ids=document_ids,
            min_quality_score=min_quality_score,
            project_name=project_name,
            department=department,
            max_security_level=max_security_level,
        )

    def search_hybrid(
        self,
        *,
        query_vector: list[float],
        query_text: str,
        top_k: int = 5,
        document_id: UUID | None = None,
        document_ids: list[UUID] | None = None,
        min_quality_score: int | None = None,
        project_name: str | None = None,
        department: str | None = None,
        max_security_level: int | None = None,
    ) -> list[dict[str, Any]]:
        """FTS 전문 검색과 pgvector 코사인 검색의 RRF 하이브리드 검색을 수행함"""
        return db_search_chunks_hybrid(
            self.settings,
            query_vector,
            query_text,
            top_k=top_k,
            document_id=document_id,
            document_ids=document_ids,
            min_quality_score=min_quality_score,
            project_name=project_name,
            department=department,
            max_security_level=max_security_level,
        )

    def delete_document(self, document_id: UUID) -> bool:
        """문서 및 하위 청크를 삭제함"""
        return db_delete_document(self.settings, document_id)

    def get_stats(self) -> dict[str, Any]:
        """PostgreSQL 시스템 통계를 반환함"""
        stats = get_system_stats(self.settings)
        stats["engine"] = "pgvector"
        return stats

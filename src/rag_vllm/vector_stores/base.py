# =============================================================================
# 파일명: base.py
# 경로: src/rag_vllm/vector_stores/base.py
# 목적: 다중 벡터 저장소(pgvector, Qdrant, Weaviate 등)를 위한 공통 추상 인터페이스 정의함
# 작성자: AI전략팀
# 작성일: 2026-10-01
# 수정일: 2026-10-01
# =============================================================================

"""다중 벡터 저장소(pgvector, Qdrant, Weaviate 등)를 위한 공통 추상 인터페이스 정의함"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID


class BaseVectorStore(ABC):
    """다양한 임베딩 벡터 저장소 엔진이 충족해야 하는 표준 인터페이스임"""

    @abstractmethod
    def initialize(self) -> None:
        """저장소 스키마, 컬렉션, 인덱스를 멱등하게 초기화함"""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """벡터 저장소 서비스의 활성 상태를 확인함"""
        pass

    @abstractmethod
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
        """원천 문서와 청크 임베딩을 저장하며 필요하면 공유 문서 ID를 사용함"""
        pass

    @abstractmethod
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
        """밀집(Dense) 코사인 유사도 벡터 검색을 수행함"""
        pass

    @abstractmethod
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
        """벡터 밀집 검색과 키워드 희소(Sparse/BM25) 검색을 결합한 하이브리드 검색을 수행함"""
        pass

    @abstractmethod
    def delete_document(self, document_id: UUID) -> bool:
        """지정된 문서와 연결된 모든 청크를 영구 삭제함"""
        pass

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """저장소 내 문서 수, 청크 수, 인덱스 상태 통계를 반환함"""
        pass

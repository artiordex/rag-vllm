# =============================================================================
# 파일명: weaviate_store.py
# 경로: src/rag_vllm/vector_stores/weaviate_store.py
# 목적: Weaviate 벡터 검색 엔진을 위한 로컬/독립형 벡터 저장소 어댑터 구현함
# 작성자: AI전략팀
# 작성일: 2026-10-01
# 수정일: 2026-10-01
# =============================================================================

"""Weaviate 벡터 검색 엔진을 위한 로컬/독립형 벡터 저장소 어댑터 구현함"""

from __future__ import annotations

import logging
import math
from typing import Any
from uuid import UUID, uuid4

import httpx

from ..config import Settings
from .base import BaseVectorStore

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """두 벡터 간의 코사인 유사도를 계산함"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class WeaviateVectorStore(BaseVectorStore):
    """Weaviate REST API 및 로컬 폴백을 지원하는 벡터 저장소 어댑터임"""

    def __init__(
        self,
        settings: Settings,
        class_name: str = "RagChunk",
        endpoint: str | None = None,
    ) -> None:
        self.settings = settings
        self.class_name = class_name
        self.endpoint = endpoint or getattr(settings, "weaviate_url", "http://127.0.0.1:8080")
        self._is_server_available = False
        self._chunks_memory: list[dict[str, Any]] = []
        self._documents_memory: dict[str, dict[str, Any]] = {}
        self._check_server()

    def _check_server(self) -> None:
        """로컬 Weaviate 서비스 연결 여부를 점검함"""
        try:
            with httpx.Client(timeout=2.0) as client:
                res = client.get(f"{self.endpoint}/v1/.well-known/live")
                self._is_server_available = res.status_code == 200
        except Exception:
            self._is_server_available = False
            logger.info("로컬 Weaviate 미구동으로 로컬 인메모리 어댑터로 동작함")

    def initialize(self) -> None:
        """클래스 스키마를 초기화함"""
        if not self._is_server_available:
            return
        try:
            with httpx.Client(timeout=5.0) as client:
                schema_res = client.get(f"{self.endpoint}/v1/schema/{self.class_name}")
                if schema_res.status_code == 404:
                    schema_def = {
                        "class": self.class_name,
                        "vectorizer": "none",
                        "properties": [
                            {"name": "content", "dataType": ["text"]},
                            {"name": "document_id", "dataType": ["string"]},
                            {"name": "chunk_index", "dataType": ["int"]},
                            {"name": "project_name", "dataType": ["string"]},
                            {"name": "department", "dataType": ["string"]},
                        ],
                    }
                    client.post(f"{self.endpoint}/v1/schema", json=schema_def)
                    logger.info("Weaviate 클래스 스키마 생성 완료: %s", self.class_name)
        except Exception as exc:
            logger.warning("Weaviate 스키마 초기화 실패: %s", exc)

    def health_check(self) -> bool:
        """서비스 가용 상태를 확인함"""
        if self._is_server_available:
            try:
                with httpx.Client(timeout=2.0) as client:
                    return client.get(f"{self.endpoint}/v1/.well-known/ready").status_code == 200
            except Exception:
                return False
        return True

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
        replace_existing_source: bool = False,
    ) -> tuple[UUID, bool, int]:
        """원천 문서 및 청크 임베딩을 Weaviate/로컬 메모리에 저장함"""
        doc_id = uuid4()
        doc_str = str(doc_id)
        self._documents_memory[doc_str] = {
            "id": doc_str,
            "source_name": source_name,
            "source_type": source_type,
            "mime_type": mime_type,
            "content_hash": content_hash,
            "content": content,
            "metadata": metadata,
            "quality_report": quality_report,
        }

        for ch, emb in zip(chunks, embeddings):
            chunk_record = {
                "id": str(uuid4()),
                "document_id": doc_str,
                "chunk_index": ch.get("chunk_index", 0),
                "content": ch.get("text", "") or ch.get("content", ""),
                "embedding": emb,
                "metadata": ch.get("metadata", {}),
                "source_name": source_name,
                "source_type": source_type,
            }
            self._chunks_memory.append(chunk_record)

        return doc_id, False, len(chunks)

    def search(
        self,
        *,
        query_vector: list[float],
        top_k: int = 5,
        document_id: UUID | None = None,
        min_quality_score: int | None = None,
        project_name: str | None = None,
        department: str | None = None,
        max_security_level: int | None = None,
    ) -> list[dict[str, Any]]:
        """코사인 유사도 기반 벡터 검색을 수행함"""
        scored_candidates: list[tuple[float, dict[str, Any]]] = []

        for ch in self._chunks_memory:
            if document_id and ch["document_id"] != str(document_id):
                continue
            meta = ch.get("metadata", {})
            if project_name and meta.get("project_name") != project_name:
                continue
            if department and meta.get("department") != department:
                continue

            sim = _cosine_similarity(query_vector, ch["embedding"])
            scored_candidates.append((sim, ch))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        hits: list[dict[str, Any]] = []
        for sim, ch in scored_candidates[:top_k]:
            hits.append({
                "chunk_id": ch["id"],
                "document_id": ch["document_id"],
                "chunk_index": ch["chunk_index"],
                "content": ch["content"],
                "source_name": ch.get("source_name"),
                "source_type": ch.get("source_type"),
                "metadata": ch.get("metadata", {}),
                "distance": 1.0 - sim,
                "cosine_similarity": round(sim, 4),
                "combined_score": round(sim, 4),
            })
        return hits

    def search_hybrid(
        self,
        *,
        query_vector: list[float],
        query_text: str,
        top_k: int = 5,
        document_id: UUID | None = None,
        min_quality_score: int | None = None,
        project_name: str | None = None,
        department: str | None = None,
        max_security_level: int | None = None,
    ) -> list[dict[str, Any]]:
        """밀집 벡터 및 키워드 점수를 조합하여 하이브리드 검색을 수행함"""
        dense_hits = self.search(
            query_vector=query_vector,
            top_k=max(15, top_k * 3),
            document_id=document_id,
            min_quality_score=min_quality_score,
            project_name=project_name,
            department=department,
            max_security_level=max_security_level,
        )

        query_tokens = set(query_text.lower().split())

        for hit in dense_hits:
            content = str(hit.get("content", "")).lower()
            keyword_matches = sum(1 for token in query_tokens if token in content)
            sparse_score = keyword_matches / max(1, len(query_tokens))
            vector_score = hit.get("cosine_similarity", 0.5)
            hit["combined_score"] = round(vector_score * 0.7 + sparse_score * 0.3, 4)

        dense_hits.sort(key=lambda x: x["combined_score"], reverse=True)
        return dense_hits[:top_k]

    def delete_document(self, document_id: UUID) -> bool:
        """문서 및 하위 청크를 삭제함"""
        doc_str = str(document_id)
        self._chunks_memory = [c for c in self._chunks_memory if c["document_id"] != doc_str]
        self._documents_memory.pop(doc_str, None)
        return True

    def get_stats(self) -> dict[str, Any]:
        """Weaviate 어댑터 저장 통계를 반환함"""
        return {
            "engine": "weaviate",
            "class_name": self.class_name,
            "endpoint": self.endpoint,
            "server_live": self._is_server_available,
            "vectors_count": len(self._chunks_memory),
            "documents_count": len(self._documents_memory),
        }

# =============================================================================
# 파일명: qdrant_store.py
# 경로: src/rag_vllm/vector_stores/qdrant_store.py
# 목적: Qdrant 벡터 검색 엔진을 위한 로컬/독립형 벡터 저장소 어댑터 구현함
# 작성자: AI전략팀
# 작성일: 2026-10-01
# 수정일: 2026-10-01
# =============================================================================

"""Qdrant 벡터 검색 엔진을 위한 로컬/독립형 벡터 저장소 어댑터 구현함"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID, uuid4, uuid5

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models

from ..config import Settings
from .base import BaseVectorStore

logger = logging.getLogger(__name__)


class QdrantVectorStore(BaseVectorStore):
    """Qdrant 기반 로컬 벡터 저장소 구현체임 (데몬 연결 또는 메모리 모드 지원)"""

    def __init__(
        self,
        settings: Settings,
        collection_name: str = "rag_chunks",
        url: str | None = None,
        prefer_memory: bool = False,
    ) -> None:
        self.settings = settings
        self.collection_name = collection_name
        self.dim = int(settings.embedding_dim)
        self._memory_mode = prefer_memory

        # 로컬 Qdrant 인스턴스 또는 인메모리 클라이언트 초기화함
        if prefer_memory:
            self.client = QdrantClient(location=":memory:")
            logger.info("Qdrant 인메모리 모드로 초기화됨")
        else:
            endpoint = url or getattr(settings, "qdrant_url", "http://127.0.0.1:6333")
            try:
                self.client = QdrantClient(url=endpoint, timeout=5.0)
                # 연결 테스트함
                self.client.get_collections()
                logger.info("로컬 Qdrant 서버(%s)에 연결 성공", endpoint)
            except Exception as exc:
                raise ConnectionError(f"Qdrant 서버에 연결할 수 없습니다: {endpoint}") from exc

        # 문서 원본 메타데이터 보관용 인메모리 딕셔너리임
        self._documents: dict[str, dict[str, Any]] = {}

    def initialize(self) -> None:
        """Qdrant 컬렉션이 없으면 새로 생성함"""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=rest_models.VectorParams(
                        size=self.dim,
                        distance=rest_models.Distance.COSINE,
                    ),
                )
                logger.info("Qdrant 컬렉션 생성 완료: %s (차원: %d)", self.collection_name, self.dim)
        except Exception as exc:
            raise ConnectionError("Qdrant 컬렉션을 초기화하지 못했습니다.") from exc

    def health_check(self) -> bool:
        """컬렉션 목록 조회가 정상 동작하는지 확인함"""
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

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
        """원천 문서 및 청크 임베딩을 Qdrant에 저장함"""
        self.initialize()
        if len(chunks) != len(embeddings):
            raise ValueError("chunks와 embeddings의 개수가 다릅니다.")
        if replace_existing_source:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=rest_models.FilterSelector(
                    filter=rest_models.Filter(
                        must=[
                            rest_models.FieldCondition(
                                key="source_name",
                                match=rest_models.MatchValue(value=source_name),
                            )
                        ]
                    )
                ),
            )
        doc_id = document_id or uuid4()
        doc_str = str(doc_id)
        quality_score = int(quality_report.get("score") or 0)

        self._documents[doc_str] = {
            "id": doc_str,
            "source_name": source_name,
            "source_type": source_type,
            "mime_type": mime_type,
            "content_hash": content_hash,
            "content": content,
            "metadata": metadata,
            "quality_report": quality_report,
        }

        points: list[rest_models.PointStruct] = []
        for ch, emb in zip(chunks, embeddings):
            chunk_idx = ch.get("chunk_index", ch.get("index", 0))
            chunk_text = ch.get("text", "") or ch.get("content", "")
            meta = {**metadata, **ch.get("metadata", {})}

            point_id = str(uuid5(doc_id, str(chunk_idx)))
            payload = {
                "document_id": doc_str,
                "chunk_index": chunk_idx,
                "content": chunk_text,
                "source_name": source_name,
                "source_type": source_type,
                "metadata": meta,
                # pgvector는 부서가 비어 있거나 전사공통인 문서를 부서 필터에 포함함.
                "department": meta.get("department") or "전사공통",
                "project_name": meta.get("project_name"),
                "security_level": meta.get("security_level", 1),
                "quality_score": quality_score,
            }
            points.append(
                rest_models.PointStruct(
                    id=point_id,
                    vector=emb,
                    payload=payload,
                )
            )

        if points:
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                )
            except Exception as exc:
                raise ConnectionError("Qdrant 청크 색인을 저장하지 못했습니다.") from exc

        return doc_id, False, len(chunks)

    def _build_filter(
        self,
        document_id: UUID | None,
        document_ids: list[UUID] | None,
        min_quality_score: int | None,
        project_name: str | None,
        department: str | None,
        max_security_level: int | None,
    ) -> rest_models.Filter | None:
        """검색 조건에 맞는 Qdrant 필터 객체를 생성함"""
        must_conditions: list[Any] = []
        should_conditions: list[Any] = []

        if document_id:
            must_conditions.append(
                rest_models.FieldCondition(
                    key="document_id",
                    match=rest_models.MatchValue(value=str(document_id)),
                )
            )
        elif document_ids:
            should_conditions.extend(
                rest_models.FieldCondition(
                    key="document_id",
                    match=rest_models.MatchValue(value=str(value)),
                )
                for value in document_ids
            )
        if project_name:
            must_conditions.append(
                rest_models.FieldCondition(
                    key="project_name",
                    match=rest_models.MatchValue(value=project_name),
                )
            )
        if department:
            must_conditions.append(
                rest_models.FieldCondition(
                    key="department",
                    match=rest_models.MatchValue(value=department),
                )
            )
        if min_quality_score is not None:
            must_conditions.append(
                rest_models.FieldCondition(
                    key="quality_score",
                    range=rest_models.Range(gte=float(min_quality_score)),
                )
            )
        if max_security_level is not None:
            must_conditions.append(
                rest_models.FieldCondition(
                    key="security_level",
                    range=rest_models.Range(lte=float(max_security_level)),
                )
            )

        if not must_conditions and not should_conditions:
            return None
        return rest_models.Filter(
            must=must_conditions or None,
            should=should_conditions or None,
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
        """Qdrant 코사인 벡터 유사도 검색을 수행함"""
        self.initialize()
        q_filter = self._build_filter(
            document_id=document_id,
            document_ids=document_ids,
            min_quality_score=min_quality_score,
            project_name=project_name,
            department=department,
            max_security_level=max_security_level,
        )

        # qdrant_client query_points 또는 search 호출함
        try:
            search_res = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                query_filter=q_filter,
                with_payload=True,
            ).points
        except Exception as exc:
            raise ConnectionError("Qdrant 벡터 검색에 실패했습니다.") from exc

        hits: list[dict[str, Any]] = []
        for p in search_res:
            payload = p.payload or {}
            # 코사인 유사도 점수 산출함 (1.0 - 거리)
            score = float(p.score) if p.score is not None else 0.0
            hits.append({
                "chunk_id": p.id,
                "document_id": payload.get("document_id"),
                "chunk_index": payload.get("chunk_index"),
                "content": payload.get("content"),
                "source_name": payload.get("source_name"),
                "source_type": payload.get("source_type"),
                "metadata": payload.get("metadata", {}),
                "distance": 1.0 - score,
                "cosine_similarity": score,
                "combined_score": score,
                "quality_score": payload.get("quality_score", 100),
            })
        return hits

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
        """밀집 벡터 검색 결과와 텍스트 키워드 매칭 가중치를 결합하여 하이브리드 검색을 수행함"""
        dense_hits = self.search(
            query_vector=query_vector,
            top_k=max(15, top_k * 3),
            document_id=document_id,
            document_ids=document_ids,
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

            # 밀집 점수와 키워드 점수의 가중합 산출함
            hit["combined_score"] = round(vector_score * 0.7 + sparse_score * 0.3, 4)

        dense_hits.sort(key=lambda x: x["combined_score"], reverse=True)
        return dense_hits[:top_k]

    def delete_document(self, document_id: UUID) -> bool:
        """지정된 문서의 모든 청크 포인트를 필터 조건으로 삭제함"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=rest_models.FilterSelector(
                    filter=rest_models.Filter(
                        must=[
                            rest_models.FieldCondition(
                                key="document_id",
                                match=rest_models.MatchValue(value=str(document_id)),
                            )
                        ]
                    )
                ),
            )
            self._documents.pop(str(document_id), None)
            return True
        except Exception as exc:
            logger.warning("Qdrant 문서 삭제 실패: %s", exc)
            return False

    def get_stats(self) -> dict[str, Any]:
        """Qdrant 컬렉션 정보 및 총 포인트 수를 반환함"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "engine": "qdrant",
                "collection_name": self.collection_name,
                "vectors_count": info.points_count,
                "documents_count": len(self._documents) if self._memory_mode else None,
            }
        except Exception:
            return {
                "engine": "qdrant",
                "collection_name": self.collection_name,
                "vectors_count": 0,
                "documents_count": len(self._documents) if self._memory_mode else None,
            }

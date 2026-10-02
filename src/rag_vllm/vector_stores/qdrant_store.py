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
import threading
from typing import Any
from uuid import UUID, uuid4, uuid5

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models

from ..config import Settings
from ..korean_search import bm25_rerank, normalize_korean_query, reciprocal_rank_fusion, tokenize_korean
from .base import BaseVectorStore

logger = logging.getLogger(__name__)
_SPARSE_CANDIDATE_LIMIT = 2_000
_SCROLL_PAGE_SIZE = 256


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
        self._init_lock = threading.Lock()
        self._initialized = False

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
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            self._initialize_collection()
            self._initialized = True

    def _initialize_collection(self) -> None:
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            if not exists:
                try:
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=rest_models.VectorParams(
                            size=self.dim,
                            distance=rest_models.Distance.COSINE,
                        ),
                    )
                except Exception:
                    # 여러 worker가 동시에 최초 생성할 때 먼저 만든 collection을 재사용함.
                    collections = self.client.get_collections().collections
                    if not any(c.name == self.collection_name for c in collections):
                        raise
                logger.info("Qdrant 컬렉션 준비 완료: %s (차원: %d)", self.collection_name, self.dim)
            if not self._memory_mode:
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="search_terms",
                        field_schema=rest_models.PayloadSchemaType.KEYWORD,
                        wait=True,
                    )
                except Exception:
                    # 다른 worker가 같은 인덱스를 먼저 만든 경우는 정상으로 처리함.
                    payload_schema = self.client.get_collection(self.collection_name).payload_schema or {}
                    if "search_terms" not in payload_schema:
                        raise
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
            project_name = metadata.get("project_name")
            replace_filter = [
                rest_models.FieldCondition(
                    key="source_name",
                    match=rest_models.MatchValue(value=source_name),
                )
            ]
            if project_name is not None:
                replace_filter.append(
                    rest_models.FieldCondition(
                        key="project_name",
                        match=rest_models.MatchValue(value=project_name),
                    )
                )
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=rest_models.FilterSelector(
                    filter=rest_models.Filter(must=replace_filter)
                ),
            )
            for existing_document_id, document in list(self._documents.items()):
                if document.get("source_name") != source_name:
                    continue
                existing_project = (document.get("metadata") or {}).get("project_name")
                if project_name is None or existing_project == project_name:
                    self._documents.pop(existing_document_id, None)
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
                "search_terms": tokenize_korean(str(chunk_text)),
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
        search_terms: list[str] | None = None,
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
        if search_terms:
            must_conditions.append(
                rest_models.FieldCondition(
                    key="search_terms",
                    match=rest_models.MatchAny(any=search_terms),
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

    def _search_keyword_candidates(
        self,
        *,
        terms: list[str],
        document_id: UUID | None,
        document_ids: list[UUID] | None,
        min_quality_score: int | None,
        project_name: str | None,
        department: str | None,
        max_security_level: int | None,
    ) -> list[dict[str, Any]]:
        if not terms:
            return []
        query_filter = self._build_filter(
            document_id=document_id,
            document_ids=document_ids,
            min_quality_score=min_quality_score,
            project_name=project_name,
            department=department,
            max_security_level=max_security_level,
            search_terms=terms,
        )
        candidates: list[dict[str, Any]] = []
        offset: Any = None
        try:
            while len(candidates) < _SPARSE_CANDIDATE_LIMIT:
                points, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=query_filter,
                    limit=min(_SCROLL_PAGE_SIZE, _SPARSE_CANDIDATE_LIMIT - len(candidates)),
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in points:
                    payload = point.payload or {}
                    candidates.append({
                        "chunk_id": point.id,
                        "document_id": payload.get("document_id"),
                        "chunk_index": payload.get("chunk_index"),
                        "content": payload.get("content", ""),
                        "source_name": payload.get("source_name"),
                        "source_type": payload.get("source_type"),
                        "metadata": payload.get("metadata", {}),
                        "quality_score": payload.get("quality_score", 100),
                        "cosine_similarity": 0.0,
                        "combined_score": 0.0,
                    })
                if not points or offset is None:
                    break
        except Exception as exc:
            logger.warning("Qdrant 형태소 키워드 검색 실패, dense 후보로 폴백합니다: %s", type(exc).__name__)
            return []
        return candidates

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
        """dense 검색과 별도 형태소 키워드 검색을 RRF로 결합함"""
        self.initialize()
        normalized_query = normalize_korean_query(query_text)
        terms = tokenize_korean(normalized_query)[:24]
        dense_limit = max(20, top_k * 4)
        dense_hits = self.search(
            query_vector=query_vector,
            top_k=dense_limit,
            document_id=document_id,
            document_ids=document_ids,
            min_quality_score=min_quality_score,
            project_name=project_name,
            department=department,
            max_security_level=max_security_level,
        )
        keyword_hits = self._search_keyword_candidates(
            terms=terms,
            document_id=document_id,
            document_ids=document_ids,
            min_quality_score=min_quality_score,
            project_name=project_name,
            department=department,
            max_security_level=max_security_level,
        )
        candidates_by_id: dict[str, dict[str, Any]] = {}
        for hit in [*dense_hits, *keyword_hits]:
            identity = hit.get("chunk_id")
            if identity is not None:
                candidates_by_id.setdefault(str(identity), hit)
        lexical_ranked = bm25_rerank(
            normalized_query,
            list(candidates_by_id.values()),
            text_key="content",
        )
        lexical_hits = [
            hit for hit in lexical_ranked
            if float(hit.get("lexical_overlap") or 0.0) > 0.0
        ]
        return reciprocal_rank_fusion(
            [dense_hits, lexical_hits],
            top_k=top_k,
            weights=[1.0, 1.2],
        )

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

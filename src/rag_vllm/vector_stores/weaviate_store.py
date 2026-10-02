"""Weaviate 공식 v4/gRPC Python 클라이언트 기반 벡터 저장소 어댑터."""

from __future__ import annotations

import atexit
import json
import logging
import math
import re
from typing import Any
from urllib.parse import urlsplit
from uuid import UUID, uuid4, uuid5

import httpx

from ..config import Settings
from ..korean_search import bm25_rerank
from .base import BaseVectorStore

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class WeaviateVectorStore(BaseVectorStore):
    """Weaviate v4/gRPC 저장소 및 서버 미구성 시 테스트용 메모리 모드를 제공함."""

    def __init__(
        self,
        settings: Settings,
        class_name: str = "RagChunk",
        endpoint: str | None = None,
        require_server: bool = False,
    ) -> None:
        if not re.fullmatch(r"[A-Z][A-Za-z0-9_]*", class_name):
            raise ValueError("Weaviate class_name은 대문자로 시작하는 식별자여야 합니다.")
        self.settings = settings
        self.class_name = class_name
        self.endpoint = (endpoint or getattr(settings, "weaviate_url", "http://127.0.0.1:8080")).rstrip("/")
        self.require_server = require_server
        self._is_server_available = self._check_server()
        self._client: Any = None
        self._collection: Any = None
        if require_server and not self._is_server_available:
            raise ConnectionError(f"Weaviate 서버에 연결할 수 없습니다: {self.endpoint}")
        self._memory_mode = not self._is_server_available
        self._chunks_memory: list[dict[str, Any]] = []
        self._documents_memory: dict[str, dict[str, Any]] = {}
        if self._memory_mode:
            logger.info("Weaviate 테스트용 인메모리 모드로 동작함")
        else:
            try:
                self._client = self._connect_v4()
            except Exception as exc:
                raise ConnectionError(
                    "Weaviate v4 클라이언트 연결에 실패했습니다. 서버 v1.27 이상과 gRPC 포트를 확인하세요."
                ) from exc
        atexit.register(self.close)

    def _check_server(self) -> bool:
        try:
            headers = (
                {"Authorization": f"Bearer {self.settings.weaviate_api_key}"}
                if self.settings.weaviate_api_key
                else None
            )
            with httpx.Client(timeout=2.0, headers=headers) as client:
                response = client.get(f"{self.endpoint}/v1/.well-known/live")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def _connect_v4(self) -> Any:
        """REST/gRPC 포트를 분리해 공식 v4 클라이언트를 연결함"""
        import weaviate
        from weaviate.auth import AuthApiKey
        from weaviate.classes.init import AdditionalConfig, Timeout

        parts = urlsplit(self.endpoint)
        host = parts.hostname
        if not host:
            raise ValueError("WEAVIATE_URL에 호스트가 필요합니다.")
        secure = parts.scheme == "https"
        client = weaviate.connect_to_custom(
            http_host=host,
            http_port=parts.port or (443 if secure else 80),
            http_secure=secure,
            grpc_host=self.settings.weaviate_grpc_host or host,
            grpc_port=self.settings.weaviate_grpc_port,
            grpc_secure=secure,
            auth_credentials=(AuthApiKey(self.settings.weaviate_api_key) if self.settings.weaviate_api_key else None),
            additional_config=AdditionalConfig(timeout=Timeout(init=5, query=15, insert=120)),
        )
        if not client.is_ready():
            client.close()
            raise ConnectionError("Weaviate가 준비 상태가 아닙니다.")
        return client

    def initialize(self) -> None:
        if self._memory_mode:
            if self.require_server:
                raise ConnectionError(f"Weaviate 서버에 연결할 수 없습니다: {self.endpoint}")
            return

        from weaviate.classes.config import Configure, DataType, Property

        if not self._client.collections.exists(self.class_name):
            try:
                self._client.collections.create(
                    name=self.class_name,
                    vector_config=Configure.Vectors.self_provided(),
                    properties=[
                        Property(name="content", data_type=DataType.TEXT),
                        Property(name="document_id", data_type=DataType.TEXT),
                        Property(name="chunk_index", data_type=DataType.INT),
                        Property(name="project_name", data_type=DataType.TEXT),
                        Property(name="department", data_type=DataType.TEXT),
                        Property(name="security_level", data_type=DataType.INT),
                        Property(name="quality_score", data_type=DataType.INT),
                        Property(name="source_name", data_type=DataType.TEXT),
                        Property(name="source_type", data_type=DataType.TEXT),
                        Property(name="metadata_json", data_type=DataType.TEXT),
                    ],
                )
            except Exception:
                # 여러 worker가 동시에 최초 시작할 때 collection 생성 경합을 허용함.
                # 생성 실패 후 실제 컬렉션이 없으면 원래 예외를 그대로 전파함.
                if not self._client.collections.exists(self.class_name):
                    raise
        self._collection = self._client.collections.use(self.class_name)

    def health_check(self) -> bool:
        if self._memory_mode:
            return not self.require_server
        try:
            return bool(self._client.is_ready())
        except Exception:
            return False

    def close(self) -> None:
        """프로세스 종료 시 gRPC/HTTP 연결 풀을 닫음"""
        if self._client is not None:
            try:
                self._client.close()
            finally:
                self._client = None
                self._collection = None

    def _filter(
        self,
        *,
        document_id: UUID | None = None,
        document_ids: list[UUID] | None = None,
        min_quality_score: int | None = None,
        project_name: str | None = None,
        department: str | None = None,
        max_security_level: int | None = None,
        source_name: str | None = None,
    ) -> Any:
        from weaviate.classes.query import Filter

        filters: list[Any] = []
        if document_id is not None:
            filters.append(Filter.by_property("document_id").equal(str(document_id)))
        elif document_ids:
            filters.append(
                Filter.any_of([
                    Filter.by_property("document_id").equal(str(value))
                    for value in document_ids
                ])
            )
        for field, value in (
            ("project_name", project_name),
            ("department", department),
            ("source_name", source_name),
        ):
            if value is not None:
                filters.append(Filter.by_property(field).equal(value))
        if min_quality_score is not None:
            filters.append(Filter.by_property("quality_score").greater_or_equal(int(min_quality_score)))
        if max_security_level is not None:
            filters.append(Filter.by_property("security_level").less_or_equal(int(max_security_level)))
        return Filter.all_of(filters) if filters else None

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
        if len(chunks) != len(embeddings):
            raise ValueError("chunks와 embeddings의 개수가 다릅니다.")
        self.initialize()
        doc_id = document_id or uuid4()
        if replace_existing_source:
            self._delete_matching(source_name=source_name, project_name=metadata.get("project_name"))
        else:
            self.delete_document(doc_id)

        doc_str = str(doc_id)
        quality_score = int(quality_report.get("score") or 0)
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

        objects: list[tuple[dict[str, Any], list[float], UUID]] = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            index = int(chunk.get("chunk_index", chunk.get("index", 0)))
            chunk_text = str(chunk.get("text", "") or chunk.get("content", ""))
            chunk_metadata = {**metadata, **(chunk.get("metadata") or {})}
            # pgvector는 부서 미지정 및 전사공통 문서를 부서별 검색에 포함함.
            chunk_metadata["department"] = chunk_metadata.get("department") or "전사공통"
            object_id = uuid5(doc_id, str(index))
            record = {
                "id": str(object_id),
                "document_id": doc_str,
                "chunk_index": index,
                "content": chunk_text,
                "embedding": embedding,
                "metadata": chunk_metadata,
                "source_name": source_name,
                "source_type": source_type,
                "quality_score": quality_score,
            }
            if self._memory_mode:
                self._chunks_memory.append(record)
                continue

            properties = {
                "content": chunk_text,
                "document_id": doc_str,
                "chunk_index": index,
                "project_name": chunk_metadata.get("project_name") or "",
                "department": chunk_metadata["department"],
                "security_level": int(chunk_metadata.get("security_level", 1) or 1),
                "quality_score": quality_score,
                "source_name": source_name,
                "source_type": source_type,
                "metadata_json": json.dumps(chunk_metadata, ensure_ascii=False),
            }
            objects.append((properties, embedding, object_id))

        if objects:
            try:
                with self._collection.batch.fixed_size(batch_size=100, concurrent_requests=2) as batch:
                    for properties, embedding, object_id in objects:
                        batch.add_object(properties=properties, vector=embedding, uuid=object_id)
                failed = self._collection.batch.failed_objects
                if failed:
                    self._delete_matching(document_id=doc_id)
                    raise ConnectionError(f"Weaviate 청크 배치 저장이 실패했습니다 ({len(failed)}건).")
            except ConnectionError:
                raise
            except Exception as exc:
                self._delete_matching(document_id=doc_id)
                raise ConnectionError("Weaviate gRPC 청크 배치 저장이 실패했습니다.") from exc

        return doc_id, False, len(chunks)

    def _delete_matching(
        self,
        *,
        document_id: UUID | None = None,
        source_name: str | None = None,
        project_name: str | None = None,
    ) -> bool:
        if self._memory_mode:
            if document_id:
                self._chunks_memory = [c for c in self._chunks_memory if c["document_id"] != str(document_id)]
                self._documents_memory.pop(str(document_id), None)
            elif source_name:
                document_ids = {
                    c["document_id"]
                    for c in self._chunks_memory
                    if c.get("source_name") == source_name
                    and (project_name is None or (c.get("metadata") or {}).get("project_name") == project_name)
                }
                self._chunks_memory = [
                    c
                    for c in self._chunks_memory
                    if c.get("source_name") != source_name
                    or (project_name is not None and (c.get("metadata") or {}).get("project_name") != project_name)
                ]
                for doc_id in document_ids:
                    self._documents_memory.pop(doc_id, None)
            return True

        where = self._filter(document_id=document_id, source_name=source_name, project_name=project_name)
        if where is None:
            return True
        try:
            self._collection.data.delete_many(where=where)
        except Exception as exc:
            raise ConnectionError("Weaviate gRPC 문서 삭제에 실패했습니다.") from exc
        if document_id:
            self._documents_memory.pop(str(document_id), None)
        elif source_name:
            stale_ids = [
                doc_id
                for doc_id, document in self._documents_memory.items()
                if document.get("source_name") == source_name
                and (project_name is None or (document.get("metadata") or {}).get("project_name") == project_name)
            ]
            for doc_id in stale_ids:
                self._documents_memory.pop(doc_id, None)
        return True

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
        self.initialize()
        if self._memory_mode:
            scored: list[tuple[float, dict[str, Any]]] = []
            for chunk in self._chunks_memory:
                if document_id and chunk["document_id"] != str(document_id):
                    continue
                if document_ids and chunk["document_id"] not in {str(value) for value in document_ids}:
                    continue
                meta = chunk.get("metadata", {})
                if project_name and meta.get("project_name") != project_name:
                    continue
                if department and meta.get("department") != department:
                    continue
                if min_quality_score is not None and chunk.get("quality_score", 100) < min_quality_score:
                    continue
                if max_security_level is not None and int(meta.get("security_level", 1)) > max_security_level:
                    continue
                scored.append((_cosine_similarity(query_vector, chunk["embedding"]), chunk))
            scored.sort(key=lambda item: item[0], reverse=True)
            records = [(score, chunk) for score, chunk in scored[:top_k]]
            return [self._format_hit(chunk, score) for score, chunk in records]

        where = self._filter(
            document_id=document_id,
            document_ids=document_ids,
            min_quality_score=min_quality_score,
            project_name=project_name,
            department=department,
            max_security_level=max_security_level,
        )
        from weaviate.classes.query import MetadataQuery

        try:
            response = self._collection.query.near_vector(
                near_vector=query_vector,
                limit=top_k,
                filters=where,
                return_properties=[
                    "content", "document_id", "chunk_index", "project_name", "department",
                    "security_level", "quality_score", "source_name", "source_type", "metadata_json",
                ],
                return_metadata=MetadataQuery(distance=True),
            )
        except Exception as exc:
            raise ConnectionError("Weaviate gRPC 벡터 검색에 실패했습니다.") from exc
        hits: list[dict[str, Any]] = []
        for obj in response.objects:
            record = obj.properties or {}
            try:
                metadata = json.loads(record.get("metadata_json") or "{}")
            except (TypeError, json.JSONDecodeError):
                metadata = {}
            raw_distance = getattr(obj.metadata, "distance", None)
            distance = float(raw_distance) if raw_distance is not None else 1.0
            hits.append(self._format_hit({
                "id": obj.uuid,
                "document_id": record.get("document_id"),
                "chunk_index": record.get("chunk_index"),
                "content": record.get("content"),
                "metadata": metadata,
                "source_name": record.get("source_name"),
                "source_type": record.get("source_type"),
                "quality_score": record.get("quality_score", 100),
            }, max(-1.0, min(1.0, 1.0 - distance))))
        return hits

    @staticmethod
    def _format_hit(chunk: dict[str, Any], similarity: float) -> dict[str, Any]:
        return {
            "chunk_id": chunk.get("id"),
            "document_id": chunk.get("document_id"),
            "chunk_index": chunk.get("chunk_index"),
            "content": chunk.get("content"),
            "source_name": chunk.get("source_name"),
            "source_type": chunk.get("source_type"),
            "metadata": chunk.get("metadata", {}),
            "quality_score": chunk.get("quality_score", 100),
            "distance": 1.0 - similarity,
            "cosine_similarity": round(similarity, 4),
            "combined_score": round(similarity, 4),
        }

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
        if self._memory_mode:
            candidates = self.search(
                query_vector=query_vector,
                top_k=max(15, top_k * 3),
                document_id=document_id,
                document_ids=document_ids,
                min_quality_score=min_quality_score,
                project_name=project_name,
                department=department,
                max_security_level=max_security_level,
            )
            candidates = bm25_rerank(query_text, candidates, text_key="content")
            scores = [float(hit.get("bm25_score") or 0.0) for hit in candidates]
            minimum, maximum = min(scores, default=0.0), max(scores, default=0.0)
            for hit in candidates:
                lexical = float(hit.get("bm25_score") or 0.0)
                bm25_score = (lexical - minimum) / (maximum - minimum) if maximum > minimum else 0.0
                lexical = 0.7 * bm25_score + 0.3 * float(hit.get("lexical_overlap") or 0.0)
                hit["combined_score"] = round(float(hit.get("cosine_similarity") or 0.0) * 0.7 + lexical * 0.3, 4)
            candidates.sort(key=lambda item: item["combined_score"], reverse=True)
            return candidates[:top_k]

        from weaviate.classes.query import MetadataQuery

        where = self._filter(
            document_id=document_id,
            document_ids=document_ids,
            min_quality_score=min_quality_score,
            project_name=project_name,
            department=department,
            max_security_level=max_security_level,
        )
        try:
            response = self._collection.query.hybrid(
                query=query_text,
                vector=query_vector,
                alpha=0.7,
                query_properties=["content"],
                limit=top_k,
                filters=where,
                return_properties=[
                    "content", "document_id", "chunk_index", "project_name", "department",
                    "security_level", "quality_score", "source_name", "source_type", "metadata_json",
                ],
                return_metadata=MetadataQuery(distance=True, score=True),
            )
        except Exception as exc:
            raise ConnectionError("Weaviate gRPC 하이브리드 검색에 실패했습니다.") from exc

        objects = response.objects
        hits: list[dict[str, Any]] = []
        native_scores: list[float | None] = []
        for obj in objects:
            record = obj.properties or {}
            try:
                metadata = json.loads(record.get("metadata_json") or "{}")
            except (TypeError, json.JSONDecodeError):
                metadata = {}
            distance = getattr(obj.metadata, "distance", None)
            similarity = max(-1.0, min(1.0, 1.0 - float(distance))) if distance is not None else 0.0
            hit = self._format_hit({
                "id": obj.uuid,
                "document_id": record.get("document_id"),
                "chunk_index": record.get("chunk_index"),
                "content": record.get("content"),
                "metadata": metadata,
                "source_name": record.get("source_name"),
                "source_type": record.get("source_type"),
                "quality_score": record.get("quality_score", 100),
            }, similarity)
            raw_score = getattr(obj.metadata, "score", None)
            try:
                score = float(raw_score) if raw_score is not None else None
                native_scores.append(score if score is not None and math.isfinite(score) else None)
            except (TypeError, ValueError, OverflowError):
                native_scores.append(None)
            hits.append(hit)
        valid_scores = [score for score in native_scores if score is not None]
        minimum, maximum = min(valid_scores, default=0.0), max(valid_scores, default=0.0)
        for rank, (hit, score) in enumerate(zip(hits, native_scores, strict=True)):
            if score is not None and maximum > minimum:
                hit["combined_score"] = round((score - minimum) / (maximum - minimum), 4)
            else:
                # SDK/server가 hybrid score를 반환하지 않는 경우 검색 결과 순서를 보존함.
                hit["combined_score"] = round(1.0 - rank / max(1, len(hits)), 4)
        return hits

    def delete_document(self, document_id: UUID) -> bool:
        return self._delete_matching(document_id=document_id)

    def get_stats(self) -> dict[str, Any]:
        if self._memory_mode:
            return {
                "engine": "weaviate-memory",
                "class_name": self.class_name,
                "vectors_count": len(self._chunks_memory),
                "documents_count": len(self._documents_memory),
            }
        try:
            aggregate = self._collection.aggregate.over_all(total_count=True)
            count = int(aggregate.total_count or 0)
        except Exception as exc:
            raise ConnectionError("Weaviate 집계 통계를 가져오지 못했습니다.") from exc
        return {
            "engine": "weaviate",
            "class_name": self.class_name,
            "endpoint": self.endpoint,
            "server_live": self.health_check(),
            "vectors_count": count,
            "documents_count": None,
        }

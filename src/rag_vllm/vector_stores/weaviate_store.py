"""Weaviate REST/GraphQL 기반 벡터 저장소 어댑터."""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any
from uuid import UUID, uuid4, uuid5

import httpx

from ..config import Settings
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
    """벡터는 Weaviate에 저장하고, 직접 테스트용 메모리 모드를 별도로 제공함."""

    _PROPERTIES = [
        {"name": "content", "dataType": ["text"]},
        {"name": "document_id", "dataType": ["text"]},
        {"name": "chunk_index", "dataType": ["int"]},
        {"name": "project_name", "dataType": ["text"]},
        {"name": "department", "dataType": ["text"]},
        {"name": "security_level", "dataType": ["int"]},
        {"name": "quality_score", "dataType": ["int"]},
        {"name": "source_name", "dataType": ["text"]},
        {"name": "source_type", "dataType": ["text"]},
        {"name": "metadata_json", "dataType": ["text"]},
    ]

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
        if require_server and not self._is_server_available:
            raise ConnectionError(f"Weaviate 서버에 연결할 수 없습니다: {self.endpoint}")
        self._memory_mode = not self._is_server_available
        self._chunks_memory: list[dict[str, Any]] = []
        self._documents_memory: dict[str, dict[str, Any]] = {}
        if self._memory_mode:
            logger.info("Weaviate 테스트용 인메모리 모드로 동작함")

    def _check_server(self) -> bool:
        try:
            with httpx.Client(timeout=2.0) as client:
                response = client.get(f"{self.endpoint}/v1/.well-known/live")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def _request(
        self,
        method: str,
        path: str,
        *,
        allowed_statuses: tuple[int, ...] = (),
        **kwargs: Any,
    ) -> httpx.Response:
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.request(method, f"{self.endpoint}{path}", **kwargs)
            if response.status_code not in allowed_statuses:
                response.raise_for_status()
            return response
        except httpx.HTTPError as exc:
            raise ConnectionError(f"Weaviate 요청에 실패했습니다: {method} {path}") from exc

    def initialize(self) -> None:
        if self._memory_mode:
            if self.require_server:
                raise ConnectionError(f"Weaviate 서버에 연결할 수 없습니다: {self.endpoint}")
            return

        schema_response = self._request(
            "GET",
            f"/v1/schema/{self.class_name}",
            allowed_statuses=(404,),
        )
        if schema_response.status_code == 404:
            self._request(
                "POST",
                "/v1/schema",
                json={
                    "class": self.class_name,
                    "vectorizer": "none",
                    "vectorIndexConfig": {"distance": "cosine"},
                    "properties": self._PROPERTIES,
                },
            )
            return
        schema = schema_response.json()
        existing = {prop.get("name") for prop in schema.get("properties", [])}
        for prop in self._PROPERTIES:
            if prop["name"] not in existing:
                self._request("POST", f"/v1/schema/{self.class_name}/properties", json=prop)

    def health_check(self) -> bool:
        return self._check_server() if self.require_server else self._memory_mode or self._check_server()

    def _where_clause(
        self,
        *,
        document_id: UUID | None = None,
        document_ids: list[UUID] | None = None,
        min_quality_score: int | None = None,
        project_name: str | None = None,
        department: str | None = None,
        max_security_level: int | None = None,
        source_name: str | None = None,
    ) -> str:
        filters: list[str] = []
        if document_id is not None:
            document_ids = None
        elif document_ids:
            ids = ", ".join(
                "{path: [\"document_id\"], operator: Equal, valueText: "
                + json.dumps(str(value))
                + "}"
                for value in document_ids
            )
            filters.append("{operator: Or, operands: [" + ids + "]}")
        for field, value in (
            ("document_id", str(document_id) if document_id else None),
            ("project_name", project_name),
            ("department", department),
            ("source_name", source_name),
        ):
            if value is not None:
                filters.append(
                    "{path: ["
                    + json.dumps(field)
                    + "], operator: Equal, valueText: "
                    + json.dumps(value, ensure_ascii=False)
                    + "}"
                )
        if min_quality_score is not None:
            filters.append(
                f"{{path: [\"quality_score\"], operator: GreaterThanEqual, valueInt: {int(min_quality_score)}}}"
            )
        if max_security_level is not None:
            filters.append(
                f"{{path: [\"security_level\"], operator: LessThanEqual, valueInt: {int(max_security_level)}}}"
            )
        if not filters:
            return ""
        return "where: {operator: And, operands: [" + ", ".join(filters) + "]}"

    def _graphql(self, query: str) -> dict[str, Any]:
        payload = self._request("POST", "/v1/graphql", json={"query": query}).json()
        errors = payload.get("errors")
        if errors:
            raise ConnectionError("Weaviate GraphQL 검색에 실패했습니다.")
        return payload.get("data", {})

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
            self._delete_matching(source_name=source_name)
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

        objects: list[dict[str, Any]] = []
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
            objects.append({
                "class": self.class_name,
                "id": str(object_id),
                "properties": properties,
                "vector": embedding,
            })

        if objects:
            for start in range(0, len(objects), 100):
                response = self._request(
                    "POST",
                    "/v1/batch/objects",
                    json={"objects": objects[start : start + 100]},
                ).json()
                failed = []
                for item in response if isinstance(response, list) else []:
                    result = item.get("result") or {}
                    errors = result.get("errors") or {}
                    if result.get("status") == "FAILED" or errors.get("error"):
                        failed.append(item.get("id", "unknown"))
                if not isinstance(response, list) or failed:
                    raise ConnectionError(
                        f"Weaviate 청크 배치 저장이 실패했습니다 ({len(failed)}건)."
                    )

        return doc_id, False, len(chunks)

    def _delete_matching(self, *, document_id: UUID | None = None, source_name: str | None = None) -> bool:
        if self._memory_mode:
            if document_id:
                self._chunks_memory = [c for c in self._chunks_memory if c["document_id"] != str(document_id)]
                self._documents_memory.pop(str(document_id), None)
            elif source_name:
                document_ids = {
                    c["document_id"] for c in self._chunks_memory if c.get("source_name") == source_name
                }
                self._chunks_memory = [c for c in self._chunks_memory if c.get("source_name") != source_name]
                for doc_id in document_ids:
                    self._documents_memory.pop(doc_id, None)
            return True

        where = self._where_clause(document_id=document_id, source_name=source_name)
        if not where:
            return True
        query = (
            f"{{ Get {{ {self.class_name}({where} limit: 10000) "
            "{ _additional { id } } } } }"
        )
        records = self._graphql(query).get("Get", {}).get(self.class_name, [])
        for record in records:
            object_id = record.get("_additional", {}).get("id")
            if object_id:
                self._request("DELETE", f"/v1/objects/{self.class_name}/{object_id}")
        if document_id:
            self._documents_memory.pop(str(document_id), None)
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

        where = self._where_clause(
            document_id=document_id,
            document_ids=document_ids,
            min_quality_score=min_quality_score,
            project_name=project_name,
            department=department,
            max_security_level=max_security_level,
        )
        vector_literal = json.dumps([float(value) for value in query_vector], allow_nan=False)
        where_arg = f", {where}" if where else ""
        query = (
            f"{{ Get {{ {self.class_name}(nearVector: {{vector: {vector_literal}}}, "
            f"limit: {int(top_k)}{where_arg}) "
            "{ content document_id chunk_index project_name department security_level quality_score "
            "source_name source_type metadata_json _additional { id distance } } } } }"
        )
        records = self._graphql(query).get("Get", {}).get(self.class_name, [])
        hits: list[dict[str, Any]] = []
        for record in records:
            extra = record.get("_additional") or {}
            try:
                metadata = json.loads(record.get("metadata_json") or "{}")
            except (TypeError, json.JSONDecodeError):
                metadata = {}
            distance = float(extra.get("distance", 1.0))
            hits.append(self._format_hit({
                "id": extra.get("id"),
                "document_id": record.get("document_id"),
                "chunk_index": record.get("chunk_index"),
                "content": record.get("content"),
                "metadata": metadata,
                "source_name": record.get("source_name"),
                "source_type": record.get("source_type"),
                "quality_score": record.get("quality_score", 100),
            }, max(-1.0, min(1.0, 1.0 - distance))) )
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
        query_tokens = set(query_text.lower().split())
        for hit in candidates:
            content = str(hit.get("content", "")).lower()
            sparse_score = sum(token in content for token in query_tokens) / max(1, len(query_tokens))
            hit["combined_score"] = round(hit.get("cosine_similarity", 0.0) * 0.7 + sparse_score * 0.3, 4)
        candidates.sort(key=lambda item: item["combined_score"], reverse=True)
        return candidates[:top_k]

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
        query = f"{{ Aggregate {{ {self.class_name} {{ meta {{ count }} }} }} }}"
        count_rows = self._graphql(query).get("Aggregate", {}).get(self.class_name, [])
        count = int(count_rows[0].get("meta", {}).get("count", 0)) if count_rows else 0
        return {
            "engine": "weaviate",
            "class_name": self.class_name,
            "endpoint": self.endpoint,
            "server_live": self.health_check(),
            "vectors_count": count,
            "documents_count": None,
        }

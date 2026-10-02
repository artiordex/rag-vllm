"""사내 C++ IndexFlat 코어를 ctypes로 호출하는 로컬 인메모리 저장소."""

from __future__ import annotations

import ctypes
import logging
import math
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from ..config import Settings
from ..korean_search import bm25_rerank, reciprocal_rank_fusion
from .base import BaseVectorStore

logger = logging.getLogger(__name__)
_UINT32_MAX = 2**32 - 1


def _library_candidates(configured: str) -> list[Path]:
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
    env_path = os.getenv("CPP_VECTOR_ENGINE_LIBRARY", "").strip()
    if env_path and Path(env_path).expanduser() not in candidates:
        candidates.append(Path(env_path).expanduser())
    project_root = Path(__file__).resolve().parents[4]
    for filename in ("libcve_shared.so", "libcve_shared.dylib", "cve_shared.dll"):
        candidates.append(project_root / "cpp-vector-engine" / "build" / filename)
    return candidates


class CppEngineVectorStore(BaseVectorStore):
    """C++ 정확 검색 인덱스와 Python 문서 메타데이터를 결합한 저장소임

    벡터와 문서 메타데이터는 모두 현재 프로세스 메모리에만 보관함. C++ 엔진의
    삭제 API가 없어 문서 삭제 때 살아 있는 벡터를 새 인덱스로 재구축함.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.dimension = settings.embedding_dim
        self.max_vectors = settings.cpp_engine_max_vectors
        self._lock = threading.RLock()
        self._library, self._library_path = self._load_library(settings.cpp_engine_library)
        self._configure_abi()
        self._handle = self._new_index()
        self._documents: dict[str, dict[str, Any]] = {}
        self._records: list[dict[str, Any]] = []

    @staticmethod
    def _load_library(configured: str) -> tuple[Any, Path]:
        attempted: list[str] = []
        for candidate in _library_candidates(configured):
            attempted.append(str(candidate))
            if not candidate.is_file():
                continue
            try:
                return ctypes.CDLL(str(candidate)), candidate.resolve()
            except OSError as exc:
                logger.warning("C++ 벡터 엔진 라이브러리 로드 실패 (%s): %s", candidate, exc)
        raise ConnectionError(
            "cpp-vector-engine 공유 라이브러리를 찾지 못했습니다. "
            "CPP_VECTOR_ENGINE_LIBRARY를 설정하고 libcve_shared를 빌드하세요. "
            f"검색 경로: {', '.join(attempted)}"
        )

    def _configure_abi(self) -> None:
        float_pointer = ctypes.POINTER(ctypes.c_float)
        uint_pointer = ctypes.POINTER(ctypes.c_uint32)
        self._library.cve_index_create.argtypes = [ctypes.c_size_t, ctypes.c_int]
        self._library.cve_index_create.restype = ctypes.c_void_p
        self._library.cve_index_destroy.argtypes = [ctypes.c_void_p]
        self._library.cve_index_destroy.restype = None
        self._library.cve_index_add.argtypes = [
            ctypes.c_void_p,
            float_pointer,
            ctypes.c_size_t,
            uint_pointer,
        ]
        self._library.cve_index_add.restype = ctypes.c_int
        self._library.cve_index_search.argtypes = [
            ctypes.c_void_p,
            float_pointer,
            ctypes.c_size_t,
            ctypes.c_size_t,
            uint_pointer,
            float_pointer,
        ]
        self._library.cve_index_search.restype = ctypes.c_size_t
        self._library.cve_index_size.argtypes = [ctypes.c_void_p]
        self._library.cve_index_size.restype = ctypes.c_size_t
        self._library.cve_last_error.argtypes = []
        self._library.cve_last_error.restype = ctypes.c_char_p

    def _new_index(self) -> int:
        # MetricType::Cosine is 1 in cve/common.hpp.
        handle = self._library.cve_index_create(self.dimension, 1)
        if not handle:
            raise ConnectionError(self._last_error("C++ 벡터 인덱스 생성 실패"))
        return int(handle)

    @staticmethod
    def _security_level(metadata: dict[str, Any]) -> int:
        try:
            return int(metadata.get("security_level", 1) or 1)
        except (TypeError, ValueError, OverflowError):
            return 1

    def _last_error(self, fallback: str) -> str:
        raw = self._library.cve_last_error()
        return raw.decode("utf-8", errors="replace") if raw else fallback

    def _add_native(self, handle: int, vector: list[float]) -> int:
        if len(vector) != self.dimension:
            raise ValueError(
                f"임베딩 차원({len(vector)})이 C++ 인덱스 차원({self.dimension})과 다릅니다."
            )
        if not all(math.isfinite(float(value)) for value in vector):
            raise ValueError("임베딩에는 유한한 숫자만 포함할 수 있습니다.")
        buffer = (ctypes.c_float * self.dimension)(*(float(value) for value in vector))
        out_id = ctypes.c_uint32()
        status = self._library.cve_index_add(handle, buffer, self.dimension, ctypes.byref(out_id))
        if status != 0:
            raise RuntimeError(self._last_error("C++ 벡터 추가 실패"))
        return int(out_id.value)

    def _rebuild_index(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        new_handle = self._new_index()
        rebuilt = [dict(record) for record in records]
        try:
            for vector_id, record in enumerate(rebuilt):
                assigned = self._add_native(new_handle, record["embedding"])
                if assigned != vector_id:
                    raise RuntimeError("C++ 엔진이 예상하지 않은 벡터 ID를 반환했습니다.")
                record["vector_id"] = assigned
        except Exception:
            self._library.cve_index_destroy(new_handle)
            raise
        old_handle = self._handle
        self._handle = new_handle
        self._library.cve_index_destroy(old_handle)
        return rebuilt

    def initialize(self) -> None:
        if not self.health_check():
            raise RuntimeError("C++ 벡터 인덱스가 초기화되지 않았습니다.")

    def health_check(self) -> bool:
        with self._lock:
            return bool(self._handle)

    def close(self) -> None:
        with self._lock:
            if self._handle:
                self._library.cve_index_destroy(self._handle)
                self._handle = 0

    def find_document_by_hash(
        self,
        source_name: str,
        content_hash: str,
        *,
        project_name: str | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            for document in self._documents.values():
                metadata = document.get("metadata") or {}
                if (
                    document.get("source_name") == source_name
                    and document.get("content_hash") == content_hash
                    and (project_name is None or metadata.get("project_name") == project_name)
                ):
                    return self._document_summary(document)
        return None

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
        if any(len(vector) != self.dimension for vector in embeddings):
            raise ValueError("청크 임베딩 차원이 설정된 C++ 인덱스 차원과 다릅니다.")

        with self._lock:
            duplicate = self.find_document_by_hash(
                source_name,
                content_hash,
                project_name=metadata.get("project_name"),
            )
            if duplicate is not None:
                return UUID(str(duplicate["id"])), True, int(duplicate["chunk_count"])

            doc_id = document_id or uuid4()
            project_name = metadata.get("project_name")
            removed_ids: set[str] = set()
            retained: list[dict[str, Any]] = []
            for record in self._records:
                same_document = record["document_id"] == str(doc_id)
                same_source = (
                    replace_existing_source
                    and record.get("source_name") == source_name
                    and (project_name is None or (record.get("metadata") or {}).get("project_name") == project_name)
                )
                if same_document or same_source:
                    removed_ids.add(record["document_id"])
                else:
                    retained.append(record)
            if len(retained) + len(chunks) > self.max_vectors:
                raise ValueError(
                    f"C++ 인메모리 벡터 상한({self.max_vectors})을 초과합니다. "
                    "CPP_ENGINE_MAX_VECTORS를 조정하거나 문서를 정리하세요."
                )

            document = {
                "id": str(doc_id),
                "source_name": source_name,
                "source_type": source_type,
                "mime_type": mime_type,
                "content_hash": content_hash,
                "content": content,
                "metadata": dict(metadata),
                "quality_report": dict(quality_report),
                "chunk_count": len(chunks),
                "created_at": datetime.now(UTC),
            }
            pending: list[dict[str, Any]] = []
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                chunk_metadata = {**metadata, **(chunk.get("metadata") or {})}
                chunk_metadata["department"] = chunk_metadata.get("department") or "전사공통"
                pending.append({
                    "id": str(uuid4()),
                    "document_id": str(doc_id),
                    "chunk_index": int(chunk.get("chunk_index", chunk.get("index", 0))),
                    "content": str(chunk.get("text", "") or chunk.get("content", "")),
                    "metadata": chunk_metadata,
                    "source_name": source_name,
                    "source_type": source_type,
                    "quality_score": int(quality_report.get("score") or 0),
                    "embedding": [float(value) for value in embedding],
                })

            rebuilt_records = [*retained, *pending]
            if removed_ids:
                rebuilt_records = self._rebuild_index(rebuilt_records)
            else:
                try:
                    for record in pending:
                        record["vector_id"] = self._add_native(self._handle, record["embedding"])
                except Exception:
                    self._records = self._rebuild_index(self._records)
                    raise
            self._records = rebuilt_records
            for removed_id in removed_ids:
                self._documents.pop(removed_id, None)
            self._documents[str(doc_id)] = document
            return doc_id, False, len(chunks)

    def _matching_records(
        self,
        *,
        document_id: UUID | None = None,
        document_ids: list[UUID] | None = None,
        min_quality_score: int | None = None,
        project_name: str | None = None,
        department: str | None = None,
        max_security_level: int | None = None,
    ) -> list[dict[str, Any]]:
        allowed_ids = {str(value) for value in document_ids} if document_ids else None
        matches: list[dict[str, Any]] = []
        for record in self._records:
            if document_id is not None and record["document_id"] != str(document_id):
                continue
            if allowed_ids is not None and record["document_id"] not in allowed_ids:
                continue
            metadata = record.get("metadata") or {}
            if project_name is not None and metadata.get("project_name") != project_name:
                continue
            department_value = metadata.get("department")
            if department is not None and department_value is not None and department_value not in (department, "전사공통"):
                continue
            if min_quality_score is not None and record.get("quality_score", 0) < min_quality_score:
                continue
            if max_security_level is not None and self._security_level(metadata) > max_security_level:
                continue
            matches.append(record)
        return matches

    def _search_records(
        self,
        query_vector: list[float],
        *,
        top_k: int,
        document_id: UUID | None = None,
        document_ids: list[UUID] | None = None,
        min_quality_score: int | None = None,
        project_name: str | None = None,
        department: str | None = None,
        max_security_level: int | None = None,
    ) -> list[dict[str, Any]]:
        if len(query_vector) != self.dimension:
            raise ValueError("질의 임베딩 차원이 C++ 인덱스 차원과 다릅니다.")
        if not all(math.isfinite(float(value)) for value in query_vector):
            raise ValueError("질의 임베딩에는 유한한 숫자만 포함할 수 있습니다.")
        if not self._records or top_k <= 0:
            return []
        query = (ctypes.c_float * self.dimension)(*(float(value) for value in query_vector))
        count = len(self._records)
        ids = (ctypes.c_uint32 * count)()
        distances = (ctypes.c_float * count)()
        returned = int(self._library.cve_index_search(
            self._handle, query, self.dimension, count, ids, distances,
        ))
        if returned == 0 and self._records:
            error = self._last_error("C++ 검색 결과를 가져오지 못했습니다.")
            if error != "":
                raise RuntimeError(error)
        records_by_id = {int(record["vector_id"]): record for record in self._records}
        scored: list[tuple[float, dict[str, Any]]] = []
        for offset in range(returned):
            record = records_by_id.get(int(ids[offset]))
            if record is None:
                continue
            metadata = record.get("metadata") or {}
            if document_id is not None and record["document_id"] != str(document_id):
                continue
            if document_ids is not None and record["document_id"] not in {str(value) for value in document_ids}:
                continue
            if project_name is not None and metadata.get("project_name") != project_name:
                continue
            department_value = metadata.get("department")
            if department is not None and department_value is not None and department_value not in (department, "전사공통"):
                continue
            if min_quality_score is not None and record.get("quality_score", 0) < min_quality_score:
                continue
            if max_security_level is not None and self._security_level(metadata) > max_security_level:
                continue
            similarity = max(-1.0, min(1.0, 1.0 - float(distances[offset])))
            hit = self._format_hit(record, similarity)
            scored.append((similarity, hit))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [hit for _score, hit in scored[:top_k]]

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
        with self._lock:
            return self._search_records(
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
        with self._lock:
            dense_hits = self._search_records(
                query_vector,
                top_k=len(self._records),
                document_id=document_id,
                document_ids=document_ids,
                min_quality_score=min_quality_score,
                project_name=project_name,
                department=department,
                max_security_level=max_security_level,
            )
            lexical_ranked = bm25_rerank(query_text, dense_hits, text_key="content")
            lexical_hits = [
                hit for hit in lexical_ranked
                if float(hit.get("lexical_overlap") or 0.0) > 0.0
            ]
            return reciprocal_rank_fusion(
                [dense_hits, lexical_hits],
                top_k=top_k,
                weights=[1.0, 1.2],
            )

    @staticmethod
    def _format_hit(record: dict[str, Any], similarity: float) -> dict[str, Any]:
        return {
            "chunk_id": record.get("id"),
            "document_id": record.get("document_id"),
            "chunk_index": record.get("chunk_index"),
            "content": record.get("content"),
            "source_name": record.get("source_name"),
            "source_type": record.get("source_type"),
            "metadata": record.get("metadata", {}),
            "quality_score": record.get("quality_score", 100),
            "distance": 1.0 - similarity,
            "cosine_similarity": round(similarity, 4),
            "combined_score": round(similarity, 4),
        }

    def _delete_matching(
        self,
        *,
        document_id: UUID | None = None,
        source_name: str | None = None,
        project_name: str | None = None,
    ) -> bool:
        removed_ids: set[str] = set()
        kept: list[dict[str, Any]] = []
        for record in self._records:
            match = (
                document_id is not None and record["document_id"] == str(document_id)
            ) or (
                source_name is not None
                and record.get("source_name") == source_name
                and (project_name is None or (record.get("metadata") or {}).get("project_name") == project_name)
            )
            if match:
                removed_ids.add(record["document_id"])
            else:
                kept.append(record)
        if removed_ids:
            self._records = self._rebuild_index(kept)
            for removed_id in removed_ids:
                self._documents.pop(removed_id, None)
        return True

    def delete_document(self, document_id: UUID) -> bool:
        with self._lock:
            return self._delete_matching(document_id=document_id)

    def find_document_id_by_source_scope(self, *, source_name: str, project_name: str) -> UUID | None:
        with self._lock:
            for document in self._documents.values():
                if (
                    document.get("source_name") == source_name
                    and (document.get("metadata") or {}).get("project_name") == project_name
                ):
                    return UUID(document["id"])
        return None

    def get_document_sources_by_ids(
        self,
        document_ids: list[UUID],
        *,
        project_name: str | None = None,
    ) -> dict[UUID, str]:
        allowed = {str(value) for value in document_ids}
        with self._lock:
            return {
                UUID(doc_id): document["source_name"]
                for doc_id, document in self._documents.items()
                if doc_id in allowed
                and (project_name is None or (document.get("metadata") or {}).get("project_name") == project_name)
            }

    def get_document(self, document_id: UUID) -> dict[str, Any] | None:
        with self._lock:
            document = self._documents.get(str(document_id))
            if document is None:
                return None
            return self._document_summary(document)

    def get_document_content(self, document_id: UUID) -> dict[str, Any] | None:
        with self._lock:
            document = self._documents.get(str(document_id))
            if document is None:
                return None
            return {
                "source_name": document["source_name"],
                "content": document["content"],
                "metadata": document["metadata"],
            }

    def get_document_chunks(self, document_id: UUID) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "id": record["id"],
                    "chunk_index": record["chunk_index"],
                    "content": record["content"],
                    "metadata": record["metadata"],
                }
                for record in sorted(self._records, key=lambda item: item["chunk_index"])
                if record["document_id"] == str(document_id)
            ]

    def list_documents(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        search: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        with self._lock:
            documents = list(self._documents.values())
            if search:
                query = search.strip().casefold()
                documents = [
                    document for document in documents
                    if query in document["source_name"].casefold() or query in document["content"].casefold()
                ]
            documents.sort(key=lambda document: document["created_at"], reverse=True)
            total = len(documents)
            page = documents[offset : offset + limit]
            return [self._document_summary(document) for document in page], total

    @staticmethod
    def _document_summary(document: dict[str, Any]) -> dict[str, Any]:
        return {
            key: document.get(key)
            for key in (
                "id", "source_name", "source_type", "mime_type", "metadata",
                "quality_report", "chunk_count", "created_at", "content_hash",
            )
        }

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "engine": "cpp-vector-engine",
                "library": str(self._library_path),
                "vectors_count": len(self._records),
                "documents_count": len(self._documents),
                "max_vectors": self.max_vectors,
                "dimension": self.dimension,
                "persistence": "process-memory-only",
            }

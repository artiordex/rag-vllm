"""PostgreSQL/pgvector persistence."""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID, uuid4

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .config import Settings

logger = logging.getLogger(__name__)


class DatabaseError(RuntimeError):
    """Raised when the storage layer cannot complete an operation."""


def _connect(settings: Settings, register_embedding: bool = False) -> psycopg.Connection[Any]:
    try:
        connection = psycopg.connect(
            settings.database_url,
            row_factory=dict_row,
            connect_timeout=settings.database_connect_timeout,
        )
        if register_embedding:
            register_vector(connection)
        return connection
    except Exception as exc:
        raise DatabaseError("PostgreSQL에 연결하지 못했습니다. DATABASE_URL을 확인하세요.") from exc


def init_db(settings: Settings) -> None:
    """Create the extension and the initial schema if they do not exist."""

    connection = _connect(settings)
    try:
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        connection.commit()
        register_vector(connection)
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_documents (
                id UUID PRIMARY KEY,
                source_name TEXT NOT NULL,
                source_type TEXT NOT NULL,
                mime_type TEXT,
                content_hash TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                quality_report JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        embedding_dimension = int(settings.embedding_dim)
        connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS rag_chunks (
                id BIGSERIAL PRIMARY KEY,
                document_id UUID NOT NULL REFERENCES rag_documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                embedding vector({embedding_dimension}) NOT NULL,
                UNIQUE(document_id, chunk_index)
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS rag_documents_source_hash_idx
            ON rag_documents (source_name, content_hash)
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_audit_logs (
                id BIGSERIAL PRIMARY KEY,
                query TEXT NOT NULL,
                search_mode TEXT,
                client_ip TEXT,
                hit_count INTEGER NOT NULL DEFAULT 0,
                confidence_score REAL,
                hallucination_risk TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        connection.commit()

        # HNSW is available in modern pgvector releases. Keep schema setup
        # usable with older extension versions by treating the index as optional.
        try:
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS rag_chunks_embedding_hnsw_idx
                ON rag_chunks USING hnsw (embedding vector_cosine_ops)
                """
            )
            connection.commit()
        except Exception as exc:  # pragma: no cover - depends on server extension version
            connection.rollback()
            logger.warning("HNSW 인덱스를 만들지 못했습니다. 소규모 데이터는 선형 검색으로 동작합니다: %s", exc)
    except Exception as exc:
        connection.rollback()
        raise DatabaseError("RAG 스키마를 초기화하지 못했습니다.") from exc
    finally:
        connection.close()


def check_db(settings: Settings) -> bool:
    connection = _connect(settings)
    try:
        connection.execute("SELECT 1")
        return True
    except Exception as exc:
        raise DatabaseError("PostgreSQL 상태 확인에 실패했습니다.") from exc
    finally:
        connection.close()


def find_document_by_hash(settings: Settings, source_name: str, content_hash: str) -> dict[str, Any] | None:
    connection = _connect(settings)
    try:
        return connection.execute(
            """
            SELECT d.id, d.source_name, d.source_type, d.mime_type, d.metadata,
                   d.quality_report, COUNT(c.id)::integer AS chunk_count
            FROM rag_documents d
            LEFT JOIN rag_chunks c ON c.document_id = d.id
            WHERE d.source_name = %s AND d.content_hash = %s
            GROUP BY d.id
            """,
            (source_name, content_hash),
        ).fetchone()
    except Exception as exc:
        raise DatabaseError("문서 중복 여부를 확인하지 못했습니다.") from exc
    finally:
        connection.close()


def save_document(
    settings: Settings,
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
) -> tuple[UUID, bool, int]:
    if len(chunks) != len(embeddings):
        raise ValueError("chunks와 embeddings의 개수가 다릅니다.")

    connection = _connect(settings, register_embedding=True)
    try:
        with connection.transaction():
            existing = connection.execute(
                "SELECT id FROM rag_documents WHERE source_name = %s AND content_hash = %s",
                (source_name, content_hash),
            ).fetchone()
            if existing:
                count = connection.execute(
                    "SELECT COUNT(*)::integer AS count FROM rag_chunks WHERE document_id = %s",
                    (existing["id"],),
                ).fetchone()["count"]
                return existing["id"], True, count

            document_id = uuid4()
            connection.execute(
                """
                INSERT INTO rag_documents
                    (id, source_name, source_type, mime_type, content_hash, content, metadata, quality_report)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    document_id,
                    source_name,
                    source_type,
                    mime_type,
                    content_hash,
                    content,
                    Jsonb(metadata),
                    Jsonb(quality_report),
                ),
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                connection.execute(
                    """
                    INSERT INTO rag_chunks (document_id, chunk_index, content, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        document_id,
                        chunk["index"],
                        chunk["text"],
                        Jsonb(chunk["metadata"]),
                        Vector(embedding),
                    ),
                )
            return document_id, False, len(chunks)
    except DatabaseError:
        raise
    except Exception as exc:
        raise DatabaseError("문서와 임베딩을 저장하지 못했습니다.") from exc
    finally:
        connection.close()


def get_document(settings: Settings, document_id: UUID) -> dict[str, Any] | None:
    connection = _connect(settings)
    try:
        return connection.execute(
            """
            SELECT d.id, d.source_name, d.source_type, d.mime_type, d.metadata,
                   d.quality_report, d.created_at, COUNT(c.id)::integer AS chunk_count
            FROM rag_documents d
            LEFT JOIN rag_chunks c ON c.document_id = d.id
            WHERE d.id = %s
            GROUP BY d.id
            """,
            (document_id,),
        ).fetchone()
    except Exception as exc:
        raise DatabaseError("문서를 조회하지 못했습니다.") from exc
    finally:
        connection.close()


def list_documents(
    settings: Settings,
    *,
    limit: int = 50,
    offset: int = 0,
    search: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """List documents with chunk counts, quality score, and pagination."""
    connection = _connect(settings)
    try:
        conditions: list[str] = []
        params: list[Any] = []
        if search:
            conditions.append("(d.source_name ILIKE %s OR d.content ILIKE %s)")
            pattern = f"%{search.strip()}%"
            params.extend([pattern, pattern])

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        count_query = f"SELECT COUNT(*)::integer AS total FROM rag_documents d {where_clause}"
        total = connection.execute(count_query, params).fetchone()["total"]

        list_query = f"""
            SELECT d.id, d.source_name, d.source_type, d.mime_type, d.metadata,
                   d.quality_report, d.created_at,
                   COUNT(c.id)::integer AS chunk_count
            FROM rag_documents d
            LEFT JOIN rag_chunks c ON c.document_id = d.id
            {where_clause}
            GROUP BY d.id
            ORDER BY d.created_at DESC
            LIMIT %s OFFSET %s
        """
        rows = connection.execute(list_query, [*params, limit, offset]).fetchall()
        return rows, total
    except Exception as exc:
        raise DatabaseError("문서 목록을 조회하지 못했습니다.") from exc
    finally:
        connection.close()


def delete_document(settings: Settings, document_id: UUID) -> bool:
    """Delete a document and its cascade-related chunks."""
    connection = _connect(settings)
    try:
        result = connection.execute("DELETE FROM rag_documents WHERE id = %s", (document_id,))
        connection.commit()
        return result.rowcount > 0
    except Exception as exc:
        connection.rollback()
        raise DatabaseError(f"문서 삭제 실패: {exc}") from exc
    finally:
        connection.close()


def get_document_chunks(settings: Settings, document_id: UUID) -> list[dict[str, Any]]:
    """Retrieve all chunks for a document."""
    connection = _connect(settings)
    try:
        return connection.execute(
            """
            SELECT id, chunk_index, content, metadata
            FROM rag_chunks
            WHERE document_id = %s
            ORDER BY chunk_index ASC
            """,
            (document_id,),
        ).fetchall()
    except Exception as exc:
        raise DatabaseError("문서 청크 목록 조회 실패") from exc
    finally:
        connection.close()


def get_system_stats(settings: Settings) -> dict[str, Any]:
    """Retrieve document and chunk counts."""
    connection = _connect(settings)
    try:
        doc_count = connection.execute("SELECT COUNT(*)::integer AS count FROM rag_documents").fetchone()["count"]
        chunk_count = connection.execute("SELECT COUNT(*)::integer AS count FROM rag_chunks").fetchone()["count"]
        return {
            "total_documents": doc_count,
            "total_chunks": chunk_count,
            "embedding_dimension": settings.embedding_dim,
            "embedding_model": settings.embedding_model,
        }
    except Exception as exc:
        raise DatabaseError("시스템 통계 조회 실패") from exc
    finally:
        connection.close()


def record_audit_log(
    settings: Settings,
    *,
    query: str,
    search_mode: str = "hybrid",
    client_ip: str | None = None,
    hit_count: int = 0,
    confidence_score: float | None = None,
    hallucination_risk: str | None = None,
) -> None:
    """Record an audit trail entry for compliance and internal security."""
    try:
        connection = _connect(settings)
        try:
            connection.execute(
                """
                INSERT INTO rag_audit_logs
                    (query, search_mode, client_ip, hit_count, confidence_score, hallucination_risk)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (query, search_mode, client_ip, hit_count, confidence_score, hallucination_risk),
            )
            connection.commit()
        finally:
            connection.close()
    except Exception as exc:
        logger.warning("감사 로그 기록 실패: %s", exc)


def get_audit_logs(settings: Settings, limit: int = 50) -> list[dict[str, Any]]:
    """Retrieve recent audit logs for security review."""
    connection = _connect(settings)
    try:
        rows = connection.execute(
            """
            SELECT id, query, search_mode, client_ip, hit_count, confidence_score,
                   hallucination_risk, created_at
            FROM rag_audit_logs
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (limit,),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "query": r["query"],
                "search_mode": r["search_mode"],
                "client_ip": r["client_ip"],
                "hit_count": r["hit_count"],
                "confidence_score": r["confidence_score"],
                "hallucination_risk": r["hallucination_risk"],
                "created_at": r["created_at"].isoformat() if hasattr(r["created_at"], "isoformat") else str(r["created_at"]),
            }
            for r in rows
        ]
    except Exception as exc:
        logger.warning("감사 로그 조회 실패: %s", exc)
        return []
    finally:
        connection.close()


def search_chunks(
    settings: Settings,
    embedding: list[float],
    *,
    top_k: int,
    document_id: UUID | None = None,
    min_quality_score: int | None = None,
    department: str | None = None,
    max_security_level: int | None = None,
) -> list[dict[str, Any]]:
    conditions: list[str] = []
    filter_params: list[Any] = []
    if document_id is not None:
        conditions.append("c.document_id = %s")
        filter_params.append(document_id)
    if min_quality_score is not None:
        conditions.append("COALESCE(NULLIF(d.quality_report->>'score', ''), '0')::integer >= %s")
        filter_params.append(min_quality_score)
    if department is not None:
        conditions.append("(c.metadata->>'department' IS NULL OR c.metadata->>'department' = %s OR c.metadata->>'department' = '전사공통')")
        filter_params.append(department)
    if max_security_level is not None:
        conditions.append("COALESCE(NULLIF(c.metadata->>'security_level', ''), '1')::integer <= %s")
        filter_params.append(max_security_level)
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    query = f"""
        SELECT c.id, c.document_id, d.source_name, c.chunk_index, c.content,
               c.metadata, d.quality_report,
               1 - (c.embedding <=> %s) AS score
        FROM rag_chunks c
        JOIN rag_documents d ON d.id = c.document_id
        {where_clause}
        ORDER BY c.embedding <=> %s
        LIMIT %s
    """
    vector = Vector(embedding)
    params = [vector, *filter_params, vector, top_k]

    connection = _connect(settings, register_embedding=True)
    try:
        rows = connection.execute(query, params).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            quality = row.get("quality_report") or {}
            results.append(
                {
                    "id": row["id"],
                    "document_id": row["document_id"],
                    "source_name": row["source_name"],
                    "chunk_index": row["chunk_index"],
                    "text": row["content"],
                    "metadata": row["metadata"] or {},
                    "score": float(row["score"]),
                    "quality_score": quality.get("score"),
                }
            )
        return results
    except Exception as exc:
        raise DatabaseError("벡터 검색에 실패했습니다.") from exc
    finally:
        connection.close()


def search_chunks_hybrid(
    settings: Settings,
    embedding: list[float],
    query_text: str,
    *,
    top_k: int = 5,
    document_id: UUID | None = None,
    min_quality_score: int | None = None,
    department: str | None = None,
    max_security_level: int | None = None,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Hybrid search combining dense vector similarity and sparse keyword matching using RRF."""
    # 1. Dense vector retrieval (top 20)
    dense_candidates = search_chunks(
        settings,
        embedding,
        top_k=max(20, top_k * 2),
        document_id=document_id,
        min_quality_score=min_quality_score,
        department=department,
        max_security_level=max_security_level,
    )

    # 2. Sparse keyword retrieval (top 20)
    keywords = [k.strip() for k in query_text.split() if len(k.strip()) >= 2]
    sparse_candidates: list[dict[str, Any]] = []

    if keywords:
        connection = _connect(settings)
        try:
            kw_conditions: list[str] = []
            kw_params: list[Any] = []
            for kw in keywords[:5]:
                kw_conditions.append("c.content ILIKE %s")
                kw_params.append(f"%{kw}%")

            extra_conditions: list[str] = []
            extra_params: list[Any] = []
            if document_id is not None:
                extra_conditions.append("c.document_id = %s")
                extra_params.append(document_id)
            if min_quality_score is not None:
                extra_conditions.append("COALESCE(NULLIF(d.quality_report->>'score', ''), '0')::integer >= %s")
                extra_params.append(min_quality_score)
            if department is not None:
                extra_conditions.append("(c.metadata->>'department' IS NULL OR c.metadata->>'department' = %s OR c.metadata->>'department' = '전사공통')")
                extra_params.append(department)
            if max_security_level is not None:
                extra_conditions.append("COALESCE(NULLIF(c.metadata->>'security_level', ''), '1')::integer <= %s")
                extra_params.append(max_security_level)

            where_parts = ["(" + " OR ".join(kw_conditions) + ")"]
            if extra_conditions:
                where_parts.extend(extra_conditions)
            where_sql = "WHERE " + " AND ".join(where_parts)

            sparse_query = f"""
                SELECT c.id, c.document_id, d.source_name, c.chunk_index, c.content,
                       c.metadata, d.quality_report
                FROM rag_chunks c
                JOIN rag_documents d ON d.id = c.document_id
                {where_sql}
                LIMIT 30
            """
            rows = connection.execute(sparse_query, [*kw_params, *extra_params]).fetchall()
            for row in rows:
                quality = row.get("quality_report") or {}
                # Simple keyword hit count for sparse rank
                text_lower = row["content"].lower()
                match_count = sum(1 for kw in keywords if kw.lower() in text_lower)
                sparse_candidates.append(
                    {
                        "id": row["id"],
                        "document_id": row["document_id"],
                        "source_name": row["source_name"],
                        "chunk_index": row["chunk_index"],
                        "text": row["content"],
                        "metadata": row["metadata"] or {},
                        "quality_score": quality.get("score"),
                        "raw_matches": match_count,
                    }
                )
            sparse_candidates.sort(key=lambda x: x["raw_matches"], reverse=True)
        except Exception as exc:
            logger.warning("키워드 보조 검색 실패, 벡터 검색 결과만 사용: %s", exc)
        finally:
            connection.close()

    # 3. Reciprocal Rank Fusion (RRF)
    all_chunks: dict[int, dict[str, Any]] = {}
    rrf_scores: dict[int, float] = {}

    for rank, item in enumerate(dense_candidates, start=1):
        cid = item["id"]
        all_chunks[cid] = item
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))

    for rank, item in enumerate(sparse_candidates, start=1):
        cid = item["id"]
        if cid not in all_chunks:
            all_chunks[cid] = {
                "id": item["id"],
                "document_id": item["document_id"],
                "source_name": item["source_name"],
                "chunk_index": item["chunk_index"],
                "text": item["text"],
                "metadata": item["metadata"],
                "quality_score": item["quality_score"],
                "score": 0.5,
            }
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))

    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
    final_results: list[dict[str, Any]] = []
    for cid in sorted_ids:
        chunk = all_chunks[cid]
        chunk_copy = dict(chunk)
        chunk_copy["rrf_score"] = round(rrf_scores[cid], 4)
        final_results.append(chunk_copy)

    return final_results

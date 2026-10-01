# =============================================================================
# 파일명: db.py
# 경로: src/rag_vllm/db.py
# 목적: PostgreSQL·pgvector 저장, 하이브리드 검색, 통계·감사 로그 제공함
# 작성자: AI전략팀
# 작성일: 2026-09-30
# 수정일: 2026-09-30
# =============================================================================

"""PostgreSQL·pgvector 저장, 하이브리드 검색, 통계·감사 로그 제공함"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any
from uuid import UUID, uuid4

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .config import Settings

logger = logging.getLogger(__name__)

_POOL: Any = None
_POOL_LOCK = threading.Lock()


class DatabaseError(RuntimeError):
    """저장 계층이 요청을 완료하지 못했음을 나타내는 예외임"""


class _PooledConnectionProxy:
    """풀에서 획득한 연결의 close() 호출 시 풀로 안전하게 반환하는 프록시임"""

    def __init__(self, pool: Any, conn: psycopg.Connection[Any]) -> None:
        self._pool = pool
        self._conn = conn
        self._closed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)

    def close(self) -> None:
        """연결을 풀로 반환함"""
        if not self._closed:
            self._closed = True
            try:
                self._pool.putconn(self._conn)
            except Exception:
                pass

    def __enter__(self) -> "_PooledConnectionProxy":
        self._conn.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        return self._conn.__exit__(exc_type, exc_val, exc_tb)


def get_db_pool(settings: Settings) -> Any:
    """설정된 풀 크기로 ConnectionPool 인스턴스를 반환하거나 생성함"""
    global _POOL
    if _POOL is None:
        with _POOL_LOCK:
            if _POOL is None:
                try:
                    from psycopg_pool import ConnectionPool

                    def configure_conn(conn: psycopg.Connection[Any]) -> None:
                        try:
                            register_vector(conn)
                        except Exception:
                            pass

                    _POOL = ConnectionPool(
                        settings.database_url,
                        min_size=settings.database_min_pool_size,
                        max_size=settings.database_max_pool_size,
                        timeout=settings.database_pool_timeout,
                        configure=configure_conn,
                        kwargs={
                            "row_factory": dict_row,
                            "connect_timeout": settings.database_connect_timeout,
                        },
                        open=True,
                    )
                except Exception as exc:
                    logger.warning("커넥션 풀 생성 실패, 개별 연결 모드로 동작함: %s", exc)
                    _POOL = False
    return _POOL


def close_db_pool(settings: Settings | None = None) -> None:
    """프로세스 종료 시 버퍼링된 감사 로그를 플러시하고 전역 커넥션 풀을 안전하게 닫음"""
    if settings is not None:
        try:
            flush_audit_logs(settings)
        except Exception:
            pass
    global _POOL
    with _POOL_LOCK:
        if _POOL and _POOL is not False:
            try:
                _POOL.close()
            except Exception:
                pass
            _POOL = None


def get_db_pool_stats(settings: Settings) -> dict[str, int]:
    """커넥션 풀의 연결 상태와 큐 대기 수 통계를 반환함"""
    pool = get_db_pool(settings)
    if pool and pool is not False:
        try:
            stats = pool.get_stats()
            return {
                "pool_min": int(stats.get("pool_min", 0)),
                "pool_max": int(stats.get("pool_max", 0)),
                "pool_size": int(stats.get("pool_size", 0)),
                "pool_available": int(stats.get("pool_available", 0)),
                "requests_waiting": int(stats.get("requests_waiting", 0)),
            }
        except Exception:
            pass
    return {"pool_min": 0, "pool_max": 0, "pool_size": 0, "pool_available": 0, "requests_waiting": 0}


def _connect(settings: Settings, register_embedding: bool = False) -> psycopg.Connection[Any]:
    """커넥션 풀 또는 단일 연결에서 PostgreSQL 세션을 획득함

    Args:
        settings: DB 주소와 연결 제한 시간을 포함한 애플리케이션 설정임
        register_embedding: pgvector 타입 등록 여부임

    Returns:
        psycopg.Connection[Any]: 연결된 PostgreSQL 세션임

    Raises:
        DatabaseError: 연결 또는 vector 타입 등록에 실패할 때 발생함
    """
    pool = get_db_pool(settings)
    if pool and pool is not False:
        try:
            conn = pool.getconn(timeout=settings.database_pool_timeout)
            if register_embedding:
                try:
                    register_vector(conn)
                except Exception:
                    pass
            return _PooledConnectionProxy(pool, conn)  # type: ignore[return-value]
        except Exception as exc:
            logger.warning("풀에서 연결 획득 실패, 단일 연결로 폴백함: %s", exc)

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
    """pgvector 확장과 문서·청크·감사 로그 스키마를 멱등적으로 생성함

    Caveats:
        구형 pgvector에서 HNSW 생성을 실패해도 소규모 데이터는 선형 검색으로
        계속 사용할 수 있도록 선택 기능으로 취급함
    """

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
            CREATE INDEX IF NOT EXISTS idx_rag_chunks_document_id
            ON rag_chunks (document_id)
            """
        )
        try:
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_rag_chunks_fts
                ON rag_chunks USING gin (to_tsvector('simple', content))
                """
            )
            connection.commit()
        except Exception:
            connection.rollback()
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

        # NOTE: HNSW는 확장 버전에 따라 지원되지 않을 수 있어 인덱스만 선택적으로 생성함
        try:
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS rag_chunks_embedding_hnsw_idx
                ON rag_chunks USING hnsw (embedding vector_cosine_ops)
                """
            )
            connection.commit()
        except Exception as exc:  # NOTE: 서버 pgvector 확장 버전에 따라 인덱스 생성 실패가 가능함
            connection.rollback()
            logger.warning("HNSW 인덱스를 만들지 못했습니다. 소규모 데이터는 선형 검색으로 동작합니다: %s", exc)
    except Exception as exc:
        connection.rollback()
        raise DatabaseError("RAG 스키마를 초기화하지 못했습니다.") from exc
    finally:
        connection.close()


def check_db(settings: Settings) -> bool:
    """PostgreSQL 연결 가능 여부를 짧은 쿼리로 확인함

    Returns:
        bool: `SELECT 1` 실행이 완료되면 True임

    Raises:
        DatabaseError: 연결 또는 상태 확인에 실패할 때 발생함
    """
    connection = _connect(settings)
    try:
        connection.execute("SELECT 1")
        return True
    except Exception as exc:
        raise DatabaseError("PostgreSQL 상태 확인에 실패했습니다.") from exc
    finally:
        connection.close()


def find_document_by_hash(settings: Settings, source_name: str, content_hash: str) -> dict[str, Any] | None:
    """원천 이름과 정규화 본문 해시가 같은 문서의 요약을 조회함

    Returns:
        dict[str, Any] | None: 중복 문서 요약 또는 대상이 없을 때 None임

    Raises:
        DatabaseError: 중복 조회에 실패할 때 발생함
    """
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
    replace_existing_source: bool = False,
) -> tuple[UUID, bool, int]:
    """문서와 모든 청크·임베딩을 하나의 트랜잭션으로 저장함

    Args:
        settings: DB 연결과 임베딩 차원 설정임
        source_name: 원천 파일 또는 문서 이름임
        source_type: 입력 경로를 나타내는 유형임
        mime_type: 원천 MIME 타입임
        content_hash: 정규화 본문의 중복 판별 해시임
        content: 저장할 정규화 원문임
        metadata: 문서 메타데이터임
        quality_report: 저장 시점의 품질 리포트임
        chunks: 청크 본문과 원문 위치 메타데이터 목록임
        embeddings: 청크별 정규화 임베딩 목록임
        replace_existing_source: 같은 source name의 이전 문서를 교체할지 여부임

    Returns:
        tuple[UUID, bool, int]: 문서 ID, 중복 여부, 저장·기존 청크 수임

    Raises:
        ValueError: 청크와 임베딩 개수가 다를 때 발생함
        DatabaseError: 저장 트랜잭션이 실패할 때 발생함

    Caveats:
        동일 이름과 해시의 문서는 새로 쓰지 않아 인제스트 재시도가 멱등적임
    """
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

            # NOTE: 프로젝트 문서 동기화는 같은 경로의 이전 버전을 남기지 않도록 교체함
            if replace_existing_source:
                connection.execute(
                    "DELETE FROM rag_documents WHERE source_name = %s",
                    (source_name,),
                )

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
            if chunks:
                chunk_params = [
                    (
                        document_id,
                        chunk["index"],
                        chunk["text"],
                        Jsonb(chunk["metadata"]),
                        Vector(embedding),
                    )
                    for chunk, embedding in zip(chunks, embeddings, strict=True)
                ]
                with connection.cursor() as cur:
                    cur.executemany(
                        """
                        INSERT INTO rag_chunks (document_id, chunk_index, content, metadata, embedding)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        chunk_params,
                    )
            return document_id, False, len(chunks)
    except DatabaseError:
        raise
    except Exception as exc:
        raise DatabaseError("문서와 임베딩을 저장하지 못했습니다.") from exc
    finally:
        connection.close()


def get_document(settings: Settings, document_id: UUID) -> dict[str, Any] | None:
    """문서 메타데이터와 품질 요약 및 청크 수를 조회함

    Returns:
        dict[str, Any] | None: 문서 요약 또는 대상이 없을 때 None임

    Raises:
        DatabaseError: 문서 조회에 실패할 때 발생함
    """
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


def get_document_content(settings: Settings, document_id: UUID) -> dict[str, Any] | None:
    """요약 조회와 분리해 저장된 추출 본문을 조회함

    Caveats:
        본문은 크기가 클 수 있어 일반 문서 목록 조회에서 불필요하게 읽지 않도록
        별도 함수로 유지함
    """
    connection = _connect(settings)
    try:
        return connection.execute(
            "SELECT source_name, content, metadata FROM rag_documents WHERE id = %s",
            (document_id,),
        ).fetchone()
    except Exception as exc:
        raise DatabaseError("문서 본문을 조회하지 못했습니다.") from exc
    finally:
        connection.close()


def list_documents(
    settings: Settings,
    *,
    limit: int = 50,
    offset: int = 0,
    search: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """검색어·페이지 조건을 적용해 문서 요약과 전체 건수를 조회함

    Returns:
        tuple[list[dict[str, Any]], int]: 현재 페이지 문서와 필터 기준 전체 건수임

    Caveats:
        검색어는 문서명과 저장 본문에 ILIKE로 적용되므로 대량 데이터에서는
        별도 전문 검색 인덱스가 필요함
    """
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
    """문서와 외래키 cascade 대상 청크를 삭제함

    Returns:
        bool: 실제 문서가 삭제되었으면 True임

    Raises:
        DatabaseError: 삭제 트랜잭션이 실패할 때 발생함
    """
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
    """문서에 속한 청크를 원문 순서로 조회함

    Returns:
        list[dict[str, Any]]: 청크 ID·순서·본문·메타데이터 목록임
    """
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
    """문서·청크 수와 현재 임베딩 설정을 반환함

    Returns:
        dict[str, Any]: 대시보드 상태 표시용 시스템 통계임
    """
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


_AUDIT_BUFFER: list[tuple[Any, ...]] = []
_AUDIT_LOCK = threading.Lock()
_AUDIT_BUFFER_MAX = 50


def flush_audit_logs(settings: Settings) -> None:
    """버퍼링된 감사 로그 목록을 PostgreSQL에 일괄 영속화함"""
    global _AUDIT_BUFFER
    logs_to_write: list[tuple[Any, ...]] = []
    with _AUDIT_LOCK:
        if _AUDIT_BUFFER:
            logs_to_write = list(_AUDIT_BUFFER)
            _AUDIT_BUFFER.clear()

    if not logs_to_write:
        return

    try:
        connection = _connect(settings)
        try:
            with connection.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO rag_audit_logs
                        (query, search_mode, client_ip, hit_count, confidence_score, hallucination_risk)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    logs_to_write,
                )
            connection.commit()
        finally:
            connection.close()
    except Exception as exc:
        logger.warning("감사 로그 배치 영속화 실패: %s", exc)


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
    """개인정보를 원문으로 남기지 않는 질의 식별자와 검색 결과를 버퍼에 기록함

    Caveats:
        호출자가 이미 비식별화한 query를 전달해야 하며, 50건 누적 시 또는
        조회·종료 시점에 데이터베이스에 일괄 반영함
    """
    item = (query, search_mode, client_ip, hit_count, confidence_score, hallucination_risk)
    should_flush = False
    with _AUDIT_LOCK:
        _AUDIT_BUFFER.append(item)
        if len(_AUDIT_BUFFER) >= _AUDIT_BUFFER_MAX:
            should_flush = True

    if should_flush:
        flush_audit_logs(settings)


def get_audit_logs(settings: Settings, limit: int = 50) -> list[dict[str, Any]]:
    """보안 검토용 최근 감사 로그를 조회하고 레거시 질의를 비공개 처리함

    Returns:
        list[dict[str, Any]]: 외부 API에 노출 가능한 감사 로그 목록임
    """
    flush_audit_logs(settings)
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
                "query": r["query"] if str(r["query"]).startswith("omitted;") else "legacy_query_redacted",
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
    document_ids: list[UUID] | None = None,
    min_quality_score: int | None = None,
    project_name: str | None = None,
    department: str | None = None,
    max_security_level: int | None = None,
) -> list[dict[str, Any]]:
    """문서·품질·부서·보안 조건을 적용해 dense 벡터 청크를 검색함

    Args:
        settings: DB 연결과 검색 설정임
        embedding: 질문 임베딩 벡터임
        top_k: 반환할 최대 결과 수임
        document_id: 지정하면 한 문서로 검색을 제한함
        document_ids: 지정하면 허용된 문서 UUID 목록으로 검색을 제한함
        min_quality_score: 지정하면 품질 점수 이상인 문서만 사용함
        project_name: 지정하면 해당 프로젝트 메타데이터 문서로 검색을 제한함
        department: 메타데이터 부서 필터임
        max_security_level: 허용할 최대 보안 등급임

    Returns:
        list[dict[str, Any]]: cosine 점수 내림차순 검색 결과 목록임

    Raises:
        DatabaseError: 벡터 검색에 실패할 때 발생함
    """
    conditions: list[str] = []
    filter_params: list[Any] = []
    if document_id is not None:
        conditions.append("c.document_id = %s")
        filter_params.append(document_id)
    elif document_ids:
        conditions.append("c.document_id = ANY(%s::uuid[])")
        filter_params.append(document_ids)
    if min_quality_score is not None:
        conditions.append("COALESCE(NULLIF(d.quality_report->>'score', ''), '0')::integer >= %s")
        filter_params.append(min_quality_score)
    if project_name is not None:
        conditions.append("d.metadata->>'project_name' = %s")
        filter_params.append(project_name)
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
    document_ids: list[UUID] | None = None,
    min_quality_score: int | None = None,
    project_name: str | None = None,
    department: str | None = None,
    max_security_level: int | None = None,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """dense 벡터와 키워드 결과를 Reciprocal Rank Fusion으로 결합함

    Returns:
        list[dict[str, Any]]: 두 검색 신호를 결합한 최종 후보 목록임

    Caveats:
        키워드 검색은 상위 5개 토큰을 ILIKE로 비교하는 보조 신호이며, 공식 BM25
        점수나 도메인 품질 판정으로 해석하지 않음
    """
    # NOTE: 의미 유사도 후보를 넉넉히 확보해 후속 RRF에서 재선택함
    dense_candidates = search_chunks(
        settings,
        embedding,
        top_k=max(20, top_k * 2),
        document_id=document_id,
        document_ids=document_ids,
        min_quality_score=min_quality_score,
        project_name=project_name,
        department=department,
        max_security_level=max_security_level,
    )

    # NOTE: 짧은 핵심어를 이용해 원문 일치 후보를 보강함 (FTS 인덱스 우선 및 ILIKE 폴백)
    keywords = [k.strip() for k in query_text.split() if len(k.strip()) >= 2]
    sparse_candidates: list[dict[str, Any]] = []

    if keywords:
        connection = _connect(settings)
        try:
            extra_conditions: list[str] = []
            extra_params: list[Any] = []
            if document_id is not None:
                extra_conditions.append("c.document_id = %s")
                extra_params.append(document_id)
            elif document_ids:
                extra_conditions.append("c.document_id = ANY(%s::uuid[])")
                extra_params.append(document_ids)
            if min_quality_score is not None:
                extra_conditions.append("COALESCE(NULLIF(d.quality_report->>'score', ''), '0')::integer >= %s")
                extra_params.append(min_quality_score)
            if project_name is not None:
                extra_conditions.append("d.metadata->>'project_name' = %s")
                extra_params.append(project_name)
            if department is not None:
                extra_conditions.append("(c.metadata->>'department' IS NULL OR c.metadata->>'department' = %s OR c.metadata->>'department' = '전사공통')")
                extra_params.append(department)
            if max_security_level is not None:
                extra_conditions.append("COALESCE(NULLIF(c.metadata->>'security_level', ''), '1')::integer <= %s")
                extra_params.append(max_security_level)

            # 1. PostgreSQL 전문 검색(FTS plainto_tsquery) 및 GIN 인덱스 활용 시도
            fts_query_str = " ".join(keywords[:5])
            fts_conditions = ["to_tsvector('simple', c.content) @@ plainto_tsquery('simple', %s)"]
            if extra_conditions:
                fts_conditions.extend(extra_conditions)
            fts_where_sql = "WHERE " + " AND ".join(fts_conditions)

            fts_sql = f"""
                SELECT c.id, c.document_id, d.source_name, c.chunk_index, c.content,
                       c.metadata, d.quality_report,
                       ts_rank_cd(to_tsvector('simple', c.content), plainto_tsquery('simple', %s)) AS fts_score
                FROM rag_chunks c
                JOIN rag_documents d ON d.id = c.document_id
                {fts_where_sql}
                ORDER BY fts_score DESC
                LIMIT 30
            """
            try:
                fts_rows = connection.execute(fts_sql, [fts_query_str, fts_query_str, *extra_params]).fetchall()
            except Exception:
                fts_rows = []

            if fts_rows:
                for row in fts_rows:
                    quality = row.get("quality_report") or {}
                    sparse_candidates.append(
                        {
                            "id": row["id"],
                            "document_id": row["document_id"],
                            "source_name": row["source_name"],
                            "chunk_index": row["chunk_index"],
                            "text": row["content"],
                            "metadata": row["metadata"] or {},
                            "quality_score": quality.get("score"),
                            "raw_matches": float(row.get("fts_score") or 1.0),
                        }
                    )
            else:
                # 2. FTS 결과 0건 시 기존 ILIKE 키워드 검색으로 안전하게 폴백함
                kw_conditions: list[str] = []
                kw_params: list[Any] = []
                for kw in keywords[:5]:
                    kw_conditions.append("c.content ILIKE %s")
                    kw_params.append(f"%{kw}%")

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
                            "raw_matches": float(match_count),
                        }
                    )
                sparse_candidates.sort(key=lambda x: x["raw_matches"], reverse=True)
        except Exception as exc:
            logger.warning("키워드 보조 검색 실패, 벡터 검색 결과만 사용: %s", exc)
        finally:
            connection.close()

    # NOTE: 질의 특성(정확 키워드/조항/코드 vs 서술형 질의)에 따라 dense·sparse 가중치를 동적으로 조절함
    dense_weight = 1.0
    sparse_weight = 1.0

    has_exact_quote = '"' in query_text or "'" in query_text
    has_article_no = bool(re.search(r"제[0-9]+조", query_text))
    has_specific_code = bool(re.search(r"[A-Z0-9_-]{4,}", query_text))

    if has_exact_quote or has_article_no or has_specific_code:
        sparse_weight = 1.4
        dense_weight = 0.8
    elif len(keywords) >= 6:
        dense_weight = 1.3
        sparse_weight = 0.7

    all_chunks: dict[int, dict[str, Any]] = {}
    rrf_scores: dict[int, float] = {}

    for rank, item in enumerate(dense_candidates, start=1):
        cid = item["id"]
        all_chunks[cid] = item
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (dense_weight / (rrf_k + rank))

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
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (sparse_weight / (rrf_k + rank))

    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
    final_results: list[dict[str, Any]] = []
    for cid in sorted_ids:
        chunk = all_chunks[cid]
        chunk_copy = dict(chunk)
        chunk_copy["rrf_score"] = round(rrf_scores[cid], 4)
        final_results.append(chunk_copy)

    return final_results

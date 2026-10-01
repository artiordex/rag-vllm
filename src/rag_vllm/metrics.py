# =============================================================================
# 파일명: metrics.py
# 경로: src/rag_vllm/metrics.py
# 목적: Prometheus 호환 실시간 서비스 지표 및 리소스 상태 수집 레지스트리 관리함
# 작성자: AI전략팀
# 작성일: 2026-10-01
# 수정일: 2026-10-01
# =============================================================================

"""Prometheus 호환 실시간 서비스 지표 및 리소스 상태 수집 레지스트리 관리함"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import Settings

_METRICS_LOCK = threading.Lock()
_REQUEST_COUNTS: dict[tuple[str, str, int], int] = defaultdict(int)
_REQUEST_DURATIONS: dict[tuple[str, str], float] = defaultdict(float)
_REQUEST_DURATION_COUNTS: dict[tuple[str, str], int] = defaultdict(int)


def record_http_request(method: str, endpoint: str, status_code: int, duration_seconds: float) -> None:
    """HTTP 요청 횟수와 응답 소요 시간을 스레드 안전하게 누적 기록함

    Args:
        method: HTTP 메서드(GET, POST 등)임
        endpoint: 정규화된 API 경로임
        status_code: 응답 HTTP 상태 코드임
        duration_seconds: 요청 처리에 소요된 시간(초)임
    """
    with _METRICS_LOCK:
        _REQUEST_COUNTS[(method.upper(), endpoint, status_code)] += 1
        _REQUEST_DURATIONS[(method.upper(), endpoint)] += duration_seconds
        _REQUEST_DURATION_COUNTS[(method.upper(), endpoint)] += 1


def reset_metrics() -> None:
    """테스트 격리를 위해 누적된 지표를 초기화함"""
    with _METRICS_LOCK:
        _REQUEST_COUNTS.clear()
        _REQUEST_DURATIONS.clear()
        _REQUEST_DURATION_COUNTS.clear()


def render_prometheus_metrics(settings: Settings | None = None) -> str:
    """누적된 지표 및 시스템 상태를 Prometheus 텍스트 포맷으로 직렬화함

    Args:
        settings: DB 커넥션 풀 및 시스템 통계 조회를 위한 설정 객체임

    Returns:
        str: Prometheus exposition 형식의 다중 라인 문자열임
    """
    lines: list[str] = [
        "# HELP rag_http_requests_total Total number of HTTP requests processed by endpoint and status.",
        "# TYPE rag_http_requests_total counter",
    ]

    with _METRICS_LOCK:
        for (method, endpoint, status_code), count in sorted(_REQUEST_COUNTS.items()):
            lines.append(f'rag_http_requests_total{{endpoint="{endpoint}",method="{method}",status="{status_code}"}} {count}')

        lines.extend([
            "# HELP rag_http_request_duration_seconds HTTP request duration in seconds.",
            "# TYPE rag_http_request_duration_seconds summary",
        ])
        for (method, endpoint), total_sec in sorted(_REQUEST_DURATIONS.items()):
            count = _REQUEST_DURATION_COUNTS.get((method, endpoint), 0)
            lines.append(f'rag_http_request_duration_seconds_sum{{endpoint="{endpoint}",method="{method}"}} {total_sec:.6f}')
            lines.append(f'rag_http_request_duration_seconds_count{{endpoint="{endpoint}",method="{method}"}} {count}')

    from .embeddings import get_query_embedding_cache_stats

    cache_stats = get_query_embedding_cache_stats()
    lines.extend([
        "# HELP rag_embeddings_cache_hits_total Total number of query embedding cache hits.",
        "# TYPE rag_embeddings_cache_hits_total counter",
        f"rag_embeddings_cache_hits_total {cache_stats['hits']}",
        "# HELP rag_embeddings_cache_misses_total Total number of query embedding cache misses.",
        "# TYPE rag_embeddings_cache_misses_total counter",
        f"rag_embeddings_cache_misses_total {cache_stats['misses']}",
        "# HELP rag_embeddings_cache_size Current number of items in query embedding cache.",
        "# TYPE rag_embeddings_cache_size gauge",
        f"rag_embeddings_cache_size {cache_stats['currsize']}",
    ])

    try:
        from .service import get_semantic_cache_stats

        sem_stats = get_semantic_cache_stats()
        lines.extend([
            "# HELP rag_semantic_cache_size Current number of items in LLM semantic cache.",
            "# TYPE rag_semantic_cache_size gauge",
            f"rag_semantic_cache_size {sem_stats['size']}",
        ])
    except Exception:
        pass

    if settings is not None:
        try:
            from .db import get_db_pool_stats

            pool_stats = get_db_pool_stats(settings)
            lines.extend([
                "# HELP rag_db_pool_connections Database connection pool connection counts.",
                "# TYPE rag_db_pool_connections gauge",
                f'rag_db_pool_connections{{state="min"}} {pool_stats["pool_min"]}',
                f'rag_db_pool_connections{{state="max"}} {pool_stats["pool_max"]}',
                f'rag_db_pool_connections{{state="size"}} {pool_stats["pool_size"]}',
                f'rag_db_pool_connections{{state="available"}} {pool_stats["pool_available"]}',
                f'rag_db_pool_connections{{state="requests_waiting"}} {pool_stats["requests_waiting"]}',
            ])
        except Exception:
            pass

    lines.append("")
    return "\n".join(lines)

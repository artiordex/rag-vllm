"""프로세스 간 공유가 가능한 bounded semantic answer cache."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from typing import Any
from uuid import UUID

from .config import Settings, get_settings

logger = logging.getLogger(__name__)

_MEMORY: list[dict[str, Any]] = []
_MEMORY_LOCK = threading.Lock()
_REDIS_LOCK = threading.Lock()
_REDIS_CLIENTS: dict[str, Any] = {}
_REDIS_WARNING_AT: dict[str, float] = {}
_PREFIX = "rag-vllm:semantic:v1:"
_GENERATION_KEY = "rag-vllm:semantic:generation:v1"
_MAX_CACHE_RECORD_BYTES = 512 * 1024


def _redis_client(settings: Settings) -> Any | None:
    url = settings.semantic_cache_redis_url
    if not url:
        return None
    with _REDIS_LOCK:
        client = _REDIS_CLIENTS.get(url)
        if client is None:
            try:
                import redis

                client = redis.Redis.from_url(
                    url,
                    decode_responses=True,
                    socket_connect_timeout=0.25,
                    socket_timeout=0.5,
                    health_check_interval=30,
                    max_connections=8,
                )
                _REDIS_CLIENTS[url] = client
            except ImportError:
                _warn_backend(url, "redis-py is not installed; using process-local semantic cache")
                return None
            except Exception as exc:
                _warn_backend(url, f"could not initialize Redis cache ({type(exc).__name__}); using local cache")
                return None
        return client


def _warn_backend(identity: str, message: str) -> None:
    now = time.monotonic()
    previous = _REDIS_WARNING_AT.get(identity, 0.0)
    if now - previous > 60:
        logger.warning("%s", message)
        _REDIS_WARNING_AT[identity] = now


def _scope_key(filter_key: str, generation: str) -> str:
    digest = hashlib.sha256(filter_key.encode("utf-8")).hexdigest()
    return f"{_PREFIX}{generation}:{digest}"


def _generation(client: Any) -> str:
    return str(client.get(_GENERATION_KEY) or "0")


def _cosine_dot(query: list[float], candidate: list[float]) -> float:
    if len(query) != len(candidate) or not query:
        return 0.0
    # 임베딩 제공자가 정규화하지 않았을 경우도 안전하게 코사인 유사도를 계산함.
    dot = sum(float(a) * float(b) for a, b in zip(query, candidate, strict=True))
    query_norm = sum(float(value) ** 2 for value in query) ** 0.5
    candidate_norm = sum(float(value) ** 2 for value in candidate) ** 0.5
    if query_norm == 0 or candidate_norm == 0:
        return 0.0
    return dot / (query_norm * candidate_norm)


def _copy_response(value: dict[str, Any]) -> dict[str, Any]:
    copied = json.loads(json.dumps(value, ensure_ascii=False, default=str))
    for source in copied.get("sources", []):
        if isinstance(source, dict) and isinstance(source.get("document_id"), str):
            try:
                source["document_id"] = UUID(source["document_id"])
            except ValueError:
                pass
    return copied


def _decode_entry(raw: str) -> dict[str, Any] | None:
    try:
        value = json.loads(raw)
        embedding = value.get("embedding")
        response = value.get("response")
        if (
            not isinstance(embedding, list)
            or not embedding
            or len(embedding) > 8192
            or not isinstance(response, dict)
        ):
            return None
        normalized = [float(item) for item in embedding]
        if any(not (-1e6 < item < 1e6) for item in normalized):
            return None
        return {"embedding": normalized, "response": response, "timestamp": float(value.get("timestamp", 0))}
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def find_semantic_response(
    settings: Settings,
    query_embedding: list[float],
    filter_key: str,
    *,
    threshold: float = 0.95,
) -> dict[str, Any] | None:
    """같은 보안·검색 범위의 최근 의미 캐시 항목을 조회함."""
    client = _redis_client(settings)
    if client is not None:
        try:
            generation = _generation(client)
            encoded = client.lrange(_scope_key(filter_key, generation), 0, settings.semantic_cache_max_entries - 1)
            for raw in encoded:
                entry = _decode_entry(raw)
                if entry and _cosine_dot(query_embedding, entry["embedding"]) >= threshold:
                    return _copy_response(entry["response"])
            return None
        except Exception as exc:
            _warn_backend(settings.semantic_cache_redis_url or "", f"Redis semantic cache read failed ({type(exc).__name__}); using local cache")

    now = time.time()
    with _MEMORY_LOCK:
        _MEMORY[:] = [
            entry for entry in _MEMORY
            if now - entry["timestamp"] <= settings.semantic_cache_ttl_seconds
        ]
        entries = list(reversed(_MEMORY))
    for entry in entries:
        if entry["filter_key"] == filter_key and _cosine_dot(query_embedding, entry["embedding"]) >= threshold:
            return dict(entry["response"])
    return None


def save_semantic_response(
    settings: Settings,
    query_embedding: list[float],
    response: dict[str, Any],
    filter_key: str,
) -> None:
    """생성 답변을 scope 별 용량과 TTL 한도 내에 저장함."""
    now = time.time()
    record = {
        "embedding": [float(value) for value in query_embedding],
        "response": json.loads(json.dumps(response, ensure_ascii=False, default=str)),
        "timestamp": now,
    }
    encoded_record = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    if len(encoded_record.encode("utf-8")) > _MAX_CACHE_RECORD_BYTES:
        return
    client = _redis_client(settings)
    if client is not None:
        try:
            key = _scope_key(filter_key, _generation(client))
            pipeline = client.pipeline(transaction=True)
            pipeline.lpush(key, encoded_record)
            pipeline.ltrim(key, 0, settings.semantic_cache_max_entries - 1)
            pipeline.expire(key, settings.semantic_cache_ttl_seconds)
            pipeline.execute()
            return
        except Exception as exc:
            _warn_backend(settings.semantic_cache_redis_url or "", f"Redis semantic cache write failed ({type(exc).__name__}); using local cache")

    with _MEMORY_LOCK:
        if len(_MEMORY) >= settings.semantic_cache_max_entries:
            del _MEMORY[: max(1, len(_MEMORY) - settings.semantic_cache_max_entries + 1)]
        _MEMORY.append({
            **record,
            "query": "",
            "filter_key": filter_key,
            "response": _copy_response(record["response"]),
        })


def clear_semantic_responses(settings: Settings | None = None) -> None:
    """인덱스 변경 시 로컬 및 사용 중인 Redis semantic cache를 무효화함."""
    with _MEMORY_LOCK:
        _MEMORY.clear()
    target_urls = {settings.semantic_cache_redis_url} if settings and settings.semantic_cache_redis_url else set()
    if settings is None:
        target_urls = set(_REDIS_CLIENTS)
    for url in target_urls:
        client = _redis_client(settings) if settings is not None else _REDIS_CLIENTS.get(url)
        if client is None:
            continue
        try:
            client.incr(_GENERATION_KEY)
            batch: list[str] = []
            for key in client.scan_iter(match=f"{_PREFIX}*", count=100):
                batch.append(key)
                if len(batch) >= 100:
                    client.delete(*batch)
                    batch.clear()
            if batch:
                client.delete(*batch)
        except Exception as exc:
            _warn_backend(url, f"Redis semantic cache clear failed ({type(exc).__name__})")


def semantic_cache_stats(settings: Settings | None = None) -> dict[str, int]:
    """현재 프로세스 크기와 구성된 한도를 반환함."""
    active_settings = settings or get_settings()
    now = time.time()
    with _MEMORY_LOCK:
        _MEMORY[:] = [
            entry for entry in _MEMORY
            if now - entry["timestamp"] <= active_settings.semantic_cache_ttl_seconds
        ]
        local_size = len(_MEMORY)
    return {
        "size": local_size,
        "max_size": active_settings.semantic_cache_max_entries,
        "redis_configured": int(bool(active_settings.semantic_cache_redis_url)),
    }

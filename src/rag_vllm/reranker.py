"""2-Stage Retrieval Cross-Encoder Reranker Module.

Stage 1: Hybrid pgvector cosine similarity + BM25-style keyword search (RRF)
Stage 2: Deep cross-attention reranking using BAAI/bge-reranker or heuristic fallback
"""

from __future__ import annotations

import logging
from typing import Any

from .config import Settings

logger = logging.getLogger(__name__)

_RERANKER_INSTANCE: Any = None


def get_reranker(settings: Settings) -> Any:
    """Return a singleton FlagReranker or None if disabled."""
    global _RERANKER_INSTANCE

    reranker_model = getattr(settings, "reranker_model", "BAAI/bge-reranker-base")
    use_reranker = getattr(settings, "use_reranker", False)

    if not use_reranker:
        return None

    if _RERANKER_INSTANCE is not None:
        return _RERANKER_INSTANCE

    try:
        from FlagEmbedding import FlagReranker

        logger.info("Initializing 2nd-stage FlagReranker: %s", reranker_model)
        _RERANKER_INSTANCE = FlagReranker(reranker_model, use_fp16=False)
        return _RERANKER_INSTANCE
    except Exception as exc:
        logger.warning("FlagReranker 로딩 실패 (어휘-휴리스틱 리랭커로 폴백): %s", exc)
        return None


def rerank_chunks(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int = 5,
    reranker: Any = None,
) -> list[dict[str, Any]]:
    """Rerank candidate chunks using Cross-Encoder or contextual score fusion.

    Returns the top_k most relevant chunks sorted by reranked score.
    """
    if not candidates:
        return []

    if len(candidates) <= 1:
        return candidates[:top_k]

    # Mode A: Neural Cross-Encoder Reranker
    if reranker is not None:
        try:
            pairs = [[query, c["text"]] for c in candidates]
            scores = reranker.compute_score(pairs)
            if isinstance(scores, (int, float)):
                scores = [float(scores)]

            scored_candidates = []
            for candidate, score in zip(candidates, scores, strict=False):
                item = dict(candidate)
                item["rerank_score"] = round(float(score), 4)
                item["score"] = round(float(score), 4)
                scored_candidates.append(item)

            scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
            return scored_candidates[:top_k]
        except Exception as exc:
            logger.warning("신경망 리랭킹 실패, 상호 순위 기반 정렬 유지: %s", exc)

    # Mode B: High-Precision Heuristic & Exact Term Cross-Scoring
    query_terms = [t.strip().lower() for t in query.split() if len(t.strip()) >= 2]
    rescored = []

    for c in candidates:
        item = dict(c)
        base_score = float(item.get("score") or item.get("rrf_score", 0.5))
        text_lower = item["text"].lower()

        # Bonus for exact query phrase match
        exact_phrase_bonus = 0.20 if query.lower() in text_lower else 0.0

        # Term density bonus
        term_hits = sum(1 for term in query_terms if term in text_lower)
        term_ratio = (term_hits / len(query_terms)) if query_terms else 0.0

        # Quality multiplier
        quality = item.get("quality_score") or 100
        quality_factor = quality / 100.0

        final_score = (base_score * 0.5) + (term_ratio * 0.3) + exact_phrase_bonus + (quality_factor * 0.1)
        item["rerank_score"] = round(final_score, 4)
        item["score"] = round(final_score, 4)
        rescored.append(item)

    rescored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return rescored[:top_k]

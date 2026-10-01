# =============================================================================
# 파일명: reranker.py
# 경로: src/rag_vllm/reranker.py
# 목적: 검색 후보를 Cross-Encoder 또는 경량 휴리스틱으로 재정렬함
# 작성자: AI전략팀
# 작성일: 2026-09-30
# 수정일: 2026-09-30
# =============================================================================

"""검색 후보를 Cross-Encoder 또는 경량 휴리스틱으로 재정렬함"""

from __future__ import annotations

import logging
import threading
from typing import Any

from .config import Settings

logger = logging.getLogger(__name__)

_RERANKER_INSTANCE: Any = None
_RERANKER_LOCK = threading.Lock()


def get_reranker(settings: Settings) -> Any:
    """설정이 허용할 때 FlagReranker 단일 인스턴스를 지연 생성함

    Returns:
        Any: FlagReranker 인스턴스 또는 비활성화·로딩 실패 시 None임

    Caveats:
        무거운 모델을 프로세스 전역에 캐시하므로 GPU 메모리 사용량을 고려해야 함
    """
    global _RERANKER_INSTANCE

    reranker_model = getattr(settings, "reranker_model", "BAAI/bge-reranker-base")
    use_reranker = getattr(settings, "use_reranker", False)

    if not use_reranker:
        return None

    if _RERANKER_INSTANCE is not None:
        return _RERANKER_INSTANCE

    with _RERANKER_LOCK:
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
    """검색 후보를 Cross-Encoder 또는 문맥 휴리스틱으로 재정렬함

    Args:
        query: 원래 사용자 질문임
        candidates: 1차 검색 결과 후보 목록임
        top_k: 반환할 최대 후보 수임
        reranker: 선택적 Cross-Encoder 인스턴스임

    Returns:
        list[dict[str, Any]]: 재정렬 점수 내림차순 후보 목록임

    Caveats:
        신경망 리랭커 실패나 미설정 시 검색 경로 유지를 위해 휴리스틱으로
        자동 전환함
    """
    if not candidates:
        return []

    if len(candidates) <= 1:
        return candidates[:top_k]

    # NOTE: 사용자가 무거운 모델을 활성화한 경우 2단계 신경망 재정렬을 우선함
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

    # NOTE: 모델 미사용·실패 시 API가 계속 응답하도록 정확 구문·용어·품질을 결합함
    query_terms = [t.strip().lower() for t in query.split() if len(t.strip()) >= 2]
    rescored = []

    for c in candidates:
        item = dict(c)
        base_score = float(item.get("score") or item.get("rrf_score", 0.5))
        text_lower = item["text"].lower()

        # NOTE: 질문 전체가 포함된 후보는 짧은 질의에서 정밀도를 보강함
        exact_phrase_bonus = 0.20 if query.lower() in text_lower else 0.0

        # NOTE: 질문 핵심어가 많이 포함된 후보를 보강함
        term_hits = sum(1 for term in query_terms if term in text_lower)
        term_ratio = (term_hits / len(query_terms)) if query_terms else 0.0

        # NOTE: 품질 점수가 낮은 문서가 검색 상위를 독점하지 않도록 감쇠함
        quality = item.get("quality_score") or 100
        quality_factor = quality / 100.0

        final_score = (base_score * 0.5) + (term_ratio * 0.3) + exact_phrase_bonus + (quality_factor * 0.1)
        item["rerank_score"] = round(final_score, 4)
        item["score"] = round(final_score, 4)
        rescored.append(item)

    rescored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return rescored[:top_k]

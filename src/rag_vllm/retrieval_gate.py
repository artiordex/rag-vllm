# =============================================================================
# 파일명: retrieval_gate.py
# 경로: src/rag_vllm/retrieval_gate.py
# 목적: 검색 후보의 어휘 근거를 점검하고 보수적인 검색 통과 결정을 반환함
# =============================================================================

"""보정 RAG 흐름에서 쓸 검색 관련성 게이트임.

검색·리랭커 점수는 서로 다른 의미와 범위를 가지며 보정된 확률이 아님.
따라서 이 모듈은 점수값을 결정 기준으로 쓰지 않고 질문과 후보 본문의
형태소 어휘 근거만 검사함. 이 결과는 의미적 정답성이나 출처의 신뢰성을
증명하지 않으며, 통과하지 못한 후보는 답변 근거로 쓰지 않도록 호출자가
재검색 또는 보류를 선택하는 보수적 신호임.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from .korean_search import tokenize_korean

GateAction = Literal["accept", "retry_retrieval", "abstain"]

_RANKING_FIELDS = (
    "score",
    "rerank_score",
    "rrf_score",
    "combined_score",
    "bm25_score",
    "lexical_overlap",
)


@dataclass(frozen=True, slots=True)
class RetrievalGateDecision:
    """검색 게이트 결과와 본문을 포함하지 않는 최소 근거 메타데이터임."""

    action: GateAction
    reason: str
    evidence: dict[str, Any]


def _normalize_phrase(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"[가-힣]+|[a-z0-9]+", normalized))


def evaluate_retrieval_relevance(
    query: str,
    candidates: Sequence[Mapping[str, Any]],
    *,
    text_key: str = "text",
    candidate_limit: int = 5,
) -> RetrievalGateDecision:
    """상위 후보의 질문 용어 근거를 점검하는 보수적 relevance gate임.

    `score`, `rerank_score`, `rrf_score` 등 숫자형 점수는 임계값 비교나
    관련성 판단에 사용하지 않음. 두 개 이상의 정보성 질문 토큰 중 최소
    두 개와 절반 이상이 한 후보 안에서 일치하거나 질문 문구 전체가
    포함된 경우에만 `accept`함. 나머지 검색 결과는 재검색 대상으로,
    평가할 질문 또는 후보 텍스트가 없으면 보류 대상으로 반환함.

    Args:
        query: 검색에 사용한 사용자 질문임.
        candidates: 검색/리랭킹 순서로 정렬된 후보 매핑 목록임.
        text_key: 후보 텍스트 키임. 기본값은 `text`이며 없으면 `content`를 읽음.
        candidate_limit: 어휘 근거를 살필 상위 후보 수임.

    Returns:
        RetrievalGateDecision: `accept`, `retry_retrieval`, `abstain` 중 하나와
        질문·본문 문구를 저장하지 않는 근거 메타데이터임.

    Raises:
        TypeError: query 또는 text_key가 문자열이 아니거나 candidates가 시퀀스가 아님.
        ValueError: candidate_limit이 양의 정수가 아님.
    """
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if not isinstance(text_key, str):
        raise TypeError("text_key must be a string")
    if isinstance(candidate_limit, bool) or not isinstance(candidate_limit, int) or candidate_limit < 1:
        raise ValueError("candidate_limit must be a positive integer")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes, bytearray)):
        raise TypeError("candidates must be a sequence of mappings")

    query_terms = tuple(dict.fromkeys(tokenize_korean(query)))
    base_evidence: dict[str, Any] = {
        "query_term_count": len(query_terms),
        "examined_candidate_count": 0,
        "usable_candidate_count": 0,
        "best_candidate_rank": None,
        "best_matched_term_count": 0,
        "best_term_coverage": 0.0,
        "exact_phrase_match": False,
        "exact_phrase_candidate_count": 0,
        "ranking_fields_observed": [],
        "ranking_scores_used": False,
        "candidate_evidence": [],
    }

    if not query.strip():
        return RetrievalGateDecision("abstain", "empty_query", base_evidence)
    if not query_terms:
        return RetrievalGateDecision("abstain", "no_informative_query_terms", base_evidence)

    selected = candidates[:candidate_limit]
    base_evidence["examined_candidate_count"] = len(selected)
    ranking_fields = {
        field
        for candidate in selected
        if isinstance(candidate, Mapping)
        for field in _RANKING_FIELDS
        if field in candidate
    }
    base_evidence["ranking_fields_observed"] = sorted(ranking_fields)

    normalized_query = _normalize_phrase(query)
    query_term_set = set(query_terms)
    best_matched_count = 0
    best_rank: int | None = None
    best_coverage = 0.0
    best_exact_phrase = False
    exact_phrase_candidate_count = 0
    best_signal = (0, 0)
    candidate_evidence: list[dict[str, Any]] = []
    usable_count = 0

    for rank, candidate in enumerate(selected, start=1):
        if not isinstance(candidate, Mapping):
            continue
        candidate_text = candidate.get(text_key)
        if candidate_text is None and text_key == "text":
            candidate_text = candidate.get("content")
        if not isinstance(candidate_text, str) or not candidate_text.strip():
            continue

        usable_count += 1
        matched_terms = query_term_set.intersection(tokenize_korean(candidate_text))
        matched_count = len(matched_terms)
        coverage = matched_count / len(query_term_set)
        normalized_candidate = _normalize_phrase(candidate_text)
        exact_phrase = bool(normalized_query and normalized_query in normalized_candidate)
        if exact_phrase:
            exact_phrase_candidate_count += 1
        candidate_evidence.append(
            {
                "rank": rank,
                "matched_term_count": matched_count,
                "term_coverage": round(coverage, 4),
                "exact_phrase_match": exact_phrase,
            }
        )

        signal = (int(exact_phrase), matched_count)
        if signal > best_signal:
            best_signal = signal
            best_matched_count = matched_count
            best_rank = rank
            best_coverage = coverage
            best_exact_phrase = exact_phrase

    base_evidence["usable_candidate_count"] = usable_count
    base_evidence["best_candidate_rank"] = best_rank
    base_evidence["best_matched_term_count"] = best_matched_count
    base_evidence["best_term_coverage"] = round(best_coverage, 4)
    base_evidence["exact_phrase_match"] = best_exact_phrase
    base_evidence["exact_phrase_candidate_count"] = exact_phrase_candidate_count
    base_evidence["candidate_evidence"] = candidate_evidence

    if usable_count == 0:
        if not selected:
            return RetrievalGateDecision("retry_retrieval", "no_candidates", base_evidence)
        return RetrievalGateDecision("abstain", "no_usable_candidate_text", base_evidence)

    minimum_matched_terms = max(2, math.ceil(len(query_term_set) / 2))
    if exact_phrase_candidate_count > 0 or (
        len(query_term_set) >= 2 and best_matched_count >= minimum_matched_terms
    ):
        return RetrievalGateDecision("accept", "sufficient_lexical_support", base_evidence)
    if best_matched_count == 0:
        return RetrievalGateDecision("retry_retrieval", "no_lexical_support", base_evidence)
    return RetrievalGateDecision("retry_retrieval", "partial_lexical_support", base_evidence)

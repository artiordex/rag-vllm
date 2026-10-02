# =============================================================================
# 파일명: korean_search.py
# 목적: 한국어 질의 핵심어 추출과 검색용 동의어 확장을 제공함
# =============================================================================

"""한국어 형태소 기반 검색어 정규화 도구임"""

from __future__ import annotations

import math
import re
import threading
from typing import Any

from .standardization import ADMINISTRATIVE_SYNONYMS

_KIWI_LOCK = threading.Lock()
_TOKEN_LOCK = threading.Lock()
_KIWI: Any = None
_KIWI_INITIALIZED = False
_NOUN_TAGS = {"NNG", "NNP", "NNB", "NR", "NP", "SL", "SN"}
_STOPWORDS = {
    "관련", "기준", "내용", "방법", "사항", "자료", "정보", "정도", "경우", "대해", "대한",
    "그리고", "또는", "어떻게", "무엇", "어디", "언제", "해주세요", "알려주세요",
}


def _get_kiwi():
    """Kiwi 사전을 프로세스에서 한 번만 초기화함"""
    global _KIWI, _KIWI_INITIALIZED
    if _KIWI_INITIALIZED:
        return _KIWI
    with _KIWI_LOCK:
        if _KIWI_INITIALIZED:
            return _KIWI
        try:
            from kiwipiepy import Kiwi
        except ImportError:
            _KIWI_INITIALIZED = True
            return None
        _KIWI = Kiwi()
        _KIWI_INITIALIZED = True
        return _KIWI


def tokenize_korean(text: str) -> list[str]:
    """명사·외래어·수치 형태소를 추출하고 불용어를 제외함"""
    kiwi = _get_kiwi()
    if kiwi is None:
        tokens = re.findall(r"[가-힣]{2,}|[a-zA-Z][a-zA-Z0-9_-]*|[0-9]+", text.lower())
    else:
        with _TOKEN_LOCK:
            analyzed = kiwi.tokenize(text)
        tokens = [token.form.lower() for token in analyzed if token.tag in _NOUN_TAGS]
    return [token for token in tokens if len(token) >= 2 and token not in _STOPWORDS]


def normalize_korean_query(text: str) -> str:
    """한국어 조사와 어미를 제외한 검색 토큰 및 검토된 동의어 후보를 반환함

    이 결과는 키워드 검색에만 사용하며 원 질문과 LLM 프롬프트는 변경하지 않음.
    로컬 표준어 매핑은 도메인 담당자의 검토가 필요한 검색 확장 후보임.
    """
    tokens = tokenize_korean(text)
    if not tokens:
        return text.strip()

    expansions: dict[str, set[str]] = {}
    for variant, canonical in ADMINISTRATIVE_SYNONYMS.items():
        expansions.setdefault(variant.lower(), set()).add(canonical.lower())
        expansions.setdefault(canonical.lower(), set()).add(variant.lower())

    normalized: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        for candidate in (token, *sorted(expansions.get(token, ()))):
            if candidate not in seen:
                seen.add(candidate)
                normalized.append(candidate)
    return " ".join(normalized[:24])


def bm25_rerank(
    query_text: str,
    candidates: list[dict[str, Any]],
    *,
    text_key: str = "text",
) -> list[dict[str, Any]]:
    """제한된 후보 집합을 형태소 토큰 BM25로 재정렬함"""
    if len(candidates) < 2:
        return candidates
    query_tokens = tokenize_korean(query_text)
    documents = [tokenize_korean(str(item.get(text_key, ""))) for item in candidates]
    if not query_tokens or not any(documents):
        return candidates

    try:
        from rank_bm25 import BM25Okapi

        scores = [float(score) for score in BM25Okapi(documents).get_scores(query_tokens)]
    except ImportError:
        scores = [float(sum(token in set(tokens) for token in query_tokens)) for tokens in documents]

    query_set = set(query_tokens)
    ranked = []
    for index, (item, score) in enumerate(zip(candidates, scores, strict=True)):
        copied = dict(item)
        copied["bm25_score"] = score
        overlap = len(query_set.intersection(documents[index])) / max(1, len(query_set))
        copied["lexical_overlap"] = overlap
        ranked.append((score, overlap, float(item.get("raw_matches") or 0), index, copied))
    ranked.sort(key=lambda entry: (-entry[0], -entry[1], -entry[2], entry[3]))
    return [entry[4] for entry in ranked]


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    *,
    top_k: int,
    rrf_k: int = 60,
    weights: list[float] | None = None,
) -> list[dict[str, Any]]:
    """서로 다른 순위 목록을 RRF로 합치고 전송 점수를 0~1 범위로 맞춤"""
    if top_k <= 0 or rrf_k <= 0 or not ranked_lists:
        return []
    if weights is not None and len(weights) != len(ranked_lists):
        raise ValueError("RRF weights must match the number of ranked lists")

    fused: dict[str, dict[str, Any]] = {}
    scores: dict[str, float] = {}
    for list_index, ranked in enumerate(ranked_lists):
        weight = float(weights[list_index]) if weights is not None else 1.0
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("RRF weights must be finite and non-negative")
        seen_in_list: set[str] = set()
        for rank, item in enumerate(ranked, start=1):
            identity = item.get("chunk_id", item.get("id"))
            if identity is None:
                continue
            key = str(identity)
            if key in seen_in_list:
                continue
            seen_in_list.add(key)
            target = fused.setdefault(key, dict(item))
            for name, value in item.items():
                if name not in target or target[name] is None:
                    target[name] = value
            scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank)

    if not scores:
        return []
    maximum = max(scores.values())
    ranked_keys = sorted(scores, key=lambda key: (-scores[key], key))[:top_k]
    results: list[dict[str, Any]] = []
    for key in ranked_keys:
        result = dict(fused[key])
        result["rrf_score"] = round(scores[key], 6)
        result["combined_score"] = round(scores[key] / maximum, 4) if maximum else 0.0
        results.append(result)
    return results


def classify_query_route(text: str) -> str:
    """문서 검색이 필요 없는 짧은 인사·감사 발화를 규칙으로 분류함"""
    normalized = re.sub(r"[\s.!?,~。！？]+", "", text).lower()
    if normalized in {
        "안녕", "안녕하세요", "안녕하십니까", "반가워요", "반갑습니다",
        "고마워", "고마워요", "감사합니다", "수고하세요", "너누구야", "너는누구니",
    }:
        return "conversation"
    return "retrieval"


def conversational_reply(text: str) -> str:
    """분류기가 판별한 일반 인사에 대한 정적 응답을 생성함"""
    normalized = re.sub(r"\s+", "", text)
    if "고마" in normalized or "감사" in normalized:
        return "천만에요. 문서 검색이나 품질 진단이 필요하면 말씀해 주세요."
    if "누구" in normalized:
        return "저는 사내 문서 검색과 비정형 데이터 품질 진단을 돕는 AI입니다."
    return "안녕하세요. 사내 문서 검색과 품질 진단을 도와드릴게요."


def should_use_hyde(text: str) -> bool:
    """핵심어가 적고 짧아 의미 간극이 커질 수 있는 질의를 골라냄"""
    return len(tokenize_korean(text)) <= 3 and len(text) <= 80

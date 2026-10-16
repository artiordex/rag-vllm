# =============================================================================
# 파일명: laya_gate.py
# 경로: src/rag_vllm/laya_gate.py
# 목적: 기존 CRAG 어휘 판정과 선택적 Laya 의미 판정을 안전하게 결합함
# =============================================================================

"""Shared CRAG decision path for ordinary and streaming RAG queries."""

from __future__ import annotations

from typing import Any

from .config import Settings
from .laya_client import assess_evidence_support
from .retrieval_gate import evaluate_retrieval_relevance


def evaluate_crag_gate(
    settings: Settings,
    *,
    question: str,
    planned_queries: list[str],
    candidate_lists: list[list[dict[str, Any]]],
    reranked_hits: list[dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    """각 하위 질의의 어휘 근거와 선택적 Laya 판단을 한 번에 평가함

    `shadow`는 실제 동작을 바꾸지 않고 Laya 판정을 기록함. `enforce`는
    기존 어휘 게이트가 보류한 경우에만 Laya가 높은 지지 확률을 반환하면
    답변 생성을 허용함. Laya 장애는 항상 기존 판정으로 폴백함.
    """
    query_decisions = [
        {
            "query": planned_query,
            "decision": evaluate_retrieval_relevance(planned_query, query_hits),
        }
        for planned_query, query_hits in zip(planned_queries, candidate_lists, strict=False)
    ]
    lexical_accept = bool(query_decisions) and all(
        item["decision"].action == "accept" for item in query_decisions
    )
    first_rejected = next(
        (item["decision"] for item in query_decisions if item["decision"].action != "accept"),
        None,
    )

    laya_metadata: dict[str, Any] | None = None
    mode = settings.laya_gate_mode
    should_call_laya = mode == "shadow" or (mode == "enforce" and not lexical_accept)
    support_probability: float | None = None
    laya_would_accept = False
    if should_call_laya:
        assessment = assess_evidence_support(
            settings,
            question=question,
            required_queries=planned_queries,
            candidates=reranked_hits,
        )
        support_probability = assessment.support_probability
        laya_would_accept = (
            assessment.status == "ok"
            and support_probability is not None
            and support_probability >= settings.laya_gate_min_support_probability
        )
        laya_metadata = {
            "mode": mode,
            "status": assessment.status,
            "support_probability": support_probability,
            "min_support_probability": settings.laya_gate_min_support_probability,
            "candidate_count": assessment.candidate_count,
            "elapsed_ms": assessment.elapsed_ms,
            "reason": assessment.reason,
            "would_accept": laya_would_accept,
        }
    elif mode != "off":
        laya_metadata = {
            "mode": mode,
            "status": "skipped_lexical_accept",
            "support_probability": None,
            "min_support_probability": settings.laya_gate_min_support_probability,
            "candidate_count": 0,
            "elapsed_ms": None,
            "reason": None,
            "would_accept": None,
        }

    laya_rescued = mode == "enforce" and not lexical_accept and laya_would_accept
    accepted = lexical_accept or laya_rescued
    reason = (
        "all_subqueries_supported"
        if lexical_accept
        else "laya_semantic_support"
        if laya_rescued
        else first_rejected.reason if first_rejected else "no_query_decisions"
    )
    gate: dict[str, Any] = {
        "action": "accept" if accepted else "abstain",
        "suggested_action": None if accepted or first_rejected is None else first_rejected.action,
        "reason": reason,
        "evidence": {
            "subqueries": [
                {"query": item["query"], **item["decision"].evidence}
                for item in query_decisions
            ],
        },
    }
    if mode != "off":
        gate["lexical_action"] = "accept" if lexical_accept else "abstain"
    if laya_metadata is not None:
        gate["laya"] = {**laya_metadata, "rescued_lexical_gate": laya_rescued}
    return gate, accepted

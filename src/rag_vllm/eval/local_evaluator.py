# =============================================================================
# 파일명: local_evaluator.py
# 경로: src/rag_vllm/eval/local_evaluator.py
# 목적: 외부 API 종속 없이 100% 로컬 환경에서 구동되는 RAG 정량 평가기 구현함
# 작성자: AI전략팀
# 작성일: 2026-10-01
# 수정일: 2026-10-01
# =============================================================================

"""외부 API 종속 없이 100% 로컬 환경에서 구동되는 RAG 정량 평가기 구현함"""

from __future__ import annotations

import logging
import math
import re
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RagEvaluationResult:
    """단일 질의응답 샘플에 대한 정량 평가 결과임"""

    query: str
    answer: str
    faithfulness: float  # 0.0 ~ 1.0 (문맥 충실도/근거성)
    answer_relevance: float  # 0.0 ~ 1.0 (질문 답변 적합성)
    context_precision: float  # 0.0 ~ 1.0 (검색 문맥 정밀도)
    context_recall: float | None = None  # 0.0 ~ 1.0 (정답 대비 문맥 재현율)
    hallucination_risk: str = "low"  # "low", "medium", "high"
    latency_ms: float = 0.0
    overall_score: float = 0.0  # 0.0 ~ 100.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """결과를 딕셔너리로 변환함"""
        return asdict(self)


@dataclass(slots=True)
class RagEvaluationReport:
    """평가 데이터셋 전체에 대한 집계 리포트임"""

    sample_count: int
    mean_faithfulness: float
    mean_answer_relevance: float
    mean_context_precision: float
    mean_overall_score: float
    p50_latency_ms: float
    p95_latency_ms: float
    high_hallucination_rate: float
    results: list[RagEvaluationResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """리포트를 딕셔너리로 변환함"""
        return {
            "sample_count": self.sample_count,
            "mean_faithfulness": self.mean_faithfulness,
            "mean_answer_relevance": self.mean_answer_relevance,
            "mean_context_precision": self.mean_context_precision,
            "mean_overall_score": self.mean_overall_score,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "high_hallucination_rate": self.high_hallucination_rate,
            "results": [r.to_dict() for r in self.results],
        }


class LocalRagEvaluator:
    """외부 클라우드 없이 로컬 토크나이저 및 어휘·의미 규칙으로 RAG 품질을 측정함"""

    def __init__(self, *, embedder: Any = None) -> None:
        self.embedder = embedder

    def _extract_keywords(self, text: str) -> set[str]:
        """어휘 분석을 위한 2자 이상 의미 단어를 추출함"""
        tokens = re.findall(r"[가-힣a-zA-Z0-9]{2,}", text.lower())
        # 기본 불용어 필터링함
        stopwords = {
            "있는", "없는", "대한", "관한", "위한", "통해", "따른", "있음", "없음",
            "그리고", "그러나", "따라서", "등에", "것으로", "하는", "이다", "합니다",
            "this", "that", "with", "from", "have", "been", "were", "what", "which",
        }
        return {t for t in tokens if t not in stopwords}

    def evaluate_faithfulness(self, answer: str, contexts: list[str]) -> tuple[float, str]:
        """답변 내 각 주장이 검색된 문맥에 얼마나 뒷받침되는지 측정함

        Returns:
            tuple[float, str]: 충실도 점수(0.0~1.0)와 환각 위험도("low", "medium", "high")임
        """
        if not answer or not contexts:
            return 0.0, "high"

        combined_context = " ".join(contexts).lower()
        sentences = [s.strip() for s in re.split(r"[.?!]\n|(?<=[.?!])\s+", answer) if len(s.strip()) > 5]
        if not sentences and answer.strip():
            sentences = [answer.strip()]

        if not sentences:
            return 1.0, "low"

        def is_word_supported(w: str, ctx: str) -> bool:
            """단어 및 한국어 어근/접두사가 문맥에 지지되는지 확인함"""
            if w in ctx:
                return True
            if len(w) >= 3 and w[:-1] in ctx:
                return True
            if len(w) >= 4 and w[:-2] in ctx:
                return True
            for cw in ctx.split():
                if len(w) >= 2 and len(cw) >= 2 and (w.startswith(cw[:2]) or cw.startswith(w[:2])):
                    return True
            return False

        grounded_count = 0
        for sent in sentences:
            sent_words = self._extract_keywords(sent)
            if not sent_words:
                grounded_count += 1
                continue
            # 문장 단어의 문맥 출현 비율 계산함
            supported = sum(1 for w in sent_words if is_word_supported(w, combined_context))
            if supported / len(sent_words) >= 0.35:
                grounded_count += 1

        faithfulness = round(grounded_count / len(sentences), 3)

        if faithfulness >= 0.75:
            risk = "low"
        elif faithfulness >= 0.45:
            risk = "medium"
        else:
            risk = "high"

        return faithfulness, risk

    def evaluate_answer_relevance(self, query: str, answer: str) -> float:
        """질문과 답변 간의 어휘 및 의미 일치도를 산출함"""
        if not query or not answer:
            return 0.0

        q_words = self._extract_keywords(query)
        if not q_words:
            return 0.5

        a_words = self._extract_keywords(answer)
        if not a_words:
            return 0.0

        # 자카드 유사도 및 포함 비율을 결합함
        intersection = q_words & a_words
        coverage = len(intersection) / len(q_words)
        jaccard = len(intersection) / len(q_words | a_words)

        relevance = 0.7 * coverage + 0.3 * (jaccard * 2.0)
        return min(1.0, round(relevance, 3))

    def evaluate_context_precision(self, query: str, contexts: list[str]) -> float:
        """검색된 청크들이 질문과 얼마나 관련되어 있는지 평가함"""
        if not query or not contexts:
            return 0.0

        q_words = self._extract_keywords(query)
        if not q_words:
            return 0.5

        relevant_chunks = 0
        for ctx in contexts:
            ctx_lower = ctx.lower()
            matches = sum(1 for w in q_words if w in ctx_lower)
            if matches / len(q_words) >= 0.2:
                relevant_chunks += 1

        return round(relevant_chunks / len(contexts), 3)

    def evaluate_context_recall(self, ground_truth: str | None, contexts: list[str]) -> float | None:
        """정답 텍스트에 포함된 사실 정보가 문맥에 포함되어 있는지 평가함"""
        if not ground_truth or not contexts:
            return None

        combined_context = " ".join(contexts).lower()
        gt_words = self._extract_keywords(ground_truth)
        if not gt_words:
            return 1.0

        covered = sum(1 for w in gt_words if w in combined_context)
        return round(covered / len(gt_words), 3)

    def evaluate_sample(
        self,
        *,
        query: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
        latency_ms: float = 0.0,
    ) -> RagEvaluationResult:
        """단일 RAG 샘플을 평가하여 종합 점수 및 지표를 산출함"""
        faithfulness, risk = self.evaluate_faithfulness(answer, contexts)
        answer_rel = self.evaluate_answer_relevance(query, answer)
        context_prec = self.evaluate_context_precision(query, contexts)
        context_rec = self.evaluate_context_recall(ground_truth, contexts)

        # 종합 점수 (0 ~ 100) 산정함
        # 가중치: 충실도 40%, 답변 적합성 35%, 문맥 정밀도 25%
        overall = (faithfulness * 40.0) + (answer_rel * 35.0) + (context_prec * 25.0)
        if risk == "high":
            overall = max(0.0, overall - 20.0)

        return RagEvaluationResult(
            query=query,
            answer=answer,
            faithfulness=faithfulness,
            answer_relevance=answer_rel,
            context_precision=context_prec,
            context_recall=context_rec,
            hallucination_risk=risk,
            latency_ms=latency_ms,
            overall_score=round(overall, 1),
        )

    def evaluate_batch(
        self,
        samples: list[dict[str, Any]],
    ) -> RagEvaluationReport:
        """복수 개의 RAG 테스트 샘플을 일괄 평가하고 요약 통계를 반환함"""
        if not samples:
            return RagEvaluationReport(
                sample_count=0,
                mean_faithfulness=0.0,
                mean_answer_relevance=0.0,
                mean_context_precision=0.0,
                mean_overall_score=0.0,
                p50_latency_ms=0.0,
                p95_latency_ms=0.0,
                high_hallucination_rate=0.0,
                results=[],
            )

        results: list[RagEvaluationResult] = []
        for s in samples:
            res = self.evaluate_sample(
                query=str(s.get("query") or s.get("input") or ""),
                answer=str(s.get("answer") or s.get("output") or ""),
                contexts=list(s.get("contexts") or s.get("retrieval_context") or []),
                ground_truth=s.get("ground_truth") or s.get("expected_output"),
                latency_ms=float(s.get("latency_ms", 0.0)),
            )
            results.append(res)

        latencies = sorted(r.latency_ms for r in results)
        n = len(latencies)
        p50 = latencies[int(n * 0.5)] if n > 0 else 0.0
        p95 = latencies[min(n - 1, int(n * 0.95))] if n > 0 else 0.0

        high_risk_count = sum(1 for r in results if r.hallucination_risk == "high")

        return RagEvaluationReport(
            sample_count=len(results),
            mean_faithfulness=round(statistics.mean(r.faithfulness for r in results), 3),
            mean_answer_relevance=round(statistics.mean(r.answer_relevance for r in results), 3),
            mean_context_precision=round(statistics.mean(r.context_precision for r in results), 3),
            mean_overall_score=round(statistics.mean(r.overall_score for r in results), 1),
            p50_latency_ms=round(p50, 1),
            p95_latency_ms=round(p95, 1),
            high_hallucination_rate=round(high_risk_count / len(results), 3),
            results=results,
        )

#!/usr/bin/env python3
"""실행 중인 RAG API를 실제 호출해 골든셋을 평가하고 요약 리포트를 생성함."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_vllm.eval.deepeval_adapter import evaluate_with_deepeval
from rag_vllm.eval.local_evaluator import LocalRagEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rag_eval")


def load_dataset(dataset_path: Path) -> list[dict[str, Any]]:
    """JSON/JSONL 골든셋을 읽고 평가 대상·기대 출처·가드레일 필드를 확인함"""
    if not dataset_path.exists():
        raise FileNotFoundError(f"평가 데이터셋을 찾을 수 없습니다: {dataset_path}")
    if dataset_path.suffix == ".jsonl":
        with dataset_path.open("r", encoding="utf-8") as stream:
            samples = [json.loads(line) for line in stream if line.strip()]
    else:
        with dataset_path.open("r", encoding="utf-8") as stream:
            raw = json.load(stream)
        if isinstance(raw, dict) and isinstance(raw.get("data"), list):
            samples = raw["data"]
        elif isinstance(raw, list):
            samples = raw
        else:
            samples = [raw]

    if not samples:
        raise ValueError("평가 데이터셋에 샘플이 없습니다.")
    for index, sample in enumerate(samples, start=1):
        if not isinstance(sample, dict) or not str(sample.get("query", "")).strip():
            raise ValueError(f"샘플 {index}에 비어 있지 않은 query 필드가 필요합니다.")
        if sample.get("evaluate_answer", True) and not str(sample.get("ground_truth", "")).strip():
            raise ValueError(f"샘플 {index}에 비어 있지 않은 ground_truth 필드가 필요합니다.")
        expected_sources = sample.get("expected_sources")
        if expected_sources is not None and (
            not isinstance(expected_sources, list)
            or any(not isinstance(source, str) or not source.strip() for source in expected_sources)
        ):
            raise ValueError(f"샘플 {index}의 expected_sources는 문서명 문자열 목록이어야 합니다.")
        expected_action = sample.get("expected_guardrail_action")
        if expected_action is not None and expected_action not in {"allow", "mask", "flag", "block"}:
            raise ValueError(f"샘플 {index}의 expected_guardrail_action 값이 유효하지 않습니다.")
        if not isinstance(sample.get("evaluate_answer", True), bool):
            raise ValueError(f"샘플 {index}의 evaluate_answer는 true 또는 false여야 합니다.")
    return samples


def _run_rag_sample(
    client: httpx.Client,
    api_url: str,
    api_key: str | None,
    sample: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    headers = {"X-API-Key": api_key} if api_key else {}
    started = time.perf_counter()
    response = client.post(
        f"{api_url.rstrip('/')}/query",
        json={"question": sample["query"], "use_llm": True},
        headers=headers,
    )
    response.raise_for_status()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    result = response.json()
    sources = [source for source in result.get("sources", []) if isinstance(source, dict)]
    contexts = [
        str(source.get("text", ""))
        for source in sources
        if source.get("text")
    ]
    return {
        "query": str(sample["query"]),
        "ground_truth": str(sample.get("ground_truth") or ""),
        "answer": str(result.get("answer") or ""),
        "contexts": contexts,
        "sources": [str(source.get("source_name") or "") for source in sources],
        "guardrail_action": result.get("guardrail_action"),
    }, elapsed_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="실행 중인 RAG API를 평가하는 로컬 골든셋 도구")
    parser.add_argument("--dataset", required=True, help="query와 ground_truth가 포함된 JSON/JSONL 파일")
    parser.add_argument("--api-url", default="http://127.0.0.1:11020", help="rag-vllm API 주소")
    parser.add_argument("--api-key", default=os.getenv("RAG_LAB_API_KEY"), help="설정된 경우 API 키")
    parser.add_argument("--engine", choices=["local", "deepeval"], default="local")
    parser.add_argument("--vllm-url", default=os.getenv("EVAL_JUDGE_BASE_URL", "http://127.0.0.1:11435/v1"))
    parser.add_argument("--model", default=os.getenv("EVAL_JUDGE_MODEL", "rag-vllm-model"))
    parser.add_argument("--output-report", default="data/eval_report.md")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--min-overall-score", type=float, default=None, help="미달 시 종료 코드 1 반환")
    parser.add_argument("--min-faithfulness", type=float, default=None, help="미달 시 종료 코드 1 반환")
    parser.add_argument(
        "--max-high-hallucination-rate",
        type=float,
        default=None,
        help="초과 시 종료 코드 1 반환",
    )
    parser.add_argument(
        "--min-source-hit-rate",
        type=float,
        default=None,
        help="기대 출처가 있는 샘플의 출처 Hit@k 기준; 미달 시 종료 코드 1 반환",
    )
    parser.add_argument(
        "--min-guardrail-pass-rate",
        type=float,
        default=None,
        help="expected_guardrail_action이 있는 샘플의 판정 일치율 기준",
    )
    args = parser.parse_args()
    if args.limit <= 0 or args.timeout <= 0:
        raise SystemExit("--limit과 --timeout은 0보다 커야 합니다.")
    if args.min_overall_score is not None and not 0 <= args.min_overall_score <= 100:
        raise SystemExit("--min-overall-score는 0~100 범위여야 합니다.")
    for name in (
        "min_faithfulness",
        "max_high_hallucination_rate",
        "min_source_hit_rate",
        "min_guardrail_pass_rate",
    ):
        value = getattr(args, name)
        if value is not None and not 0 <= value <= 1:
            raise SystemExit(f"--{name.replace('_', '-')}는 0~1 범위여야 합니다.")

    samples = load_dataset(Path(args.dataset))[: args.limit]
    if not samples:
        raise SystemExit("--limit 적용 후 평가할 샘플이 없습니다.")

    rows: list[dict[str, Any]] = []
    local_samples: list[dict[str, Any]] = []
    deepeval_scores: list[tuple[float, float]] = []
    retrieval_checks: list[dict[str, Any]] = []
    source_recall_checks: list[float] = []
    source_precision_checks: list[float] = []
    guardrail_checks: list[bool] = []
    with httpx.Client(timeout=args.timeout) as client:
        for index, sample in enumerate(samples, start=1):
            actual, latency_ms = _run_rag_sample(client, args.api_url, args.api_key, sample)
            row: dict[str, Any] = {"index": index, "latency_ms": round(latency_ms, 2)}
            if sample.get("evaluate_answer", True):
                local_samples.append({
                    "query": actual["query"],
                    "answer": actual["answer"],
                    "contexts": actual["contexts"],
                    "ground_truth": actual["ground_truth"],
                    "latency_ms": latency_ms,
                })
            if args.engine == "deepeval" and sample.get("evaluate_answer", True):
                metrics = evaluate_with_deepeval(
                    input_text=actual["query"],
                    actual_output=actual["answer"],
                    retrieval_context=actual["contexts"],
                    expected_output=actual["ground_truth"],
                    base_url=args.vllm_url,
                    model_name=args.model,
                )
                faithfulness = metrics.get("faithfulness", {}).get("score")
                answer_relevancy = metrics.get("answer_relevancy", {}).get("score")
                if faithfulness is None or answer_relevancy is None:
                    raise RuntimeError(f"DeepEval 샘플 {index} 계산에 실패했습니다: {metrics}")
                deepeval_scores.append((float(faithfulness), float(answer_relevancy)))
                row["faithfulness"] = float(faithfulness)
                row["answer_relevancy"] = float(answer_relevancy)
            expected_sources = sample.get("expected_sources")
            if expected_sources is not None:
                expected = {source.casefold() for source in expected_sources}
                retrieved = [source.casefold() for source in actual["sources"] if source]
                matched = expected.intersection(retrieved)
                source_hit = bool(matched) if expected else not retrieved
                source_check: dict[str, Any] = {
                    "source_hit": source_hit,
                    "expected_sources": expected_sources,
                    "retrieved_sources": actual["sources"],
                }
                if expected:
                    source_recall = len(matched) / len(expected)
                    source_precision = len(matched) / max(1, len(set(retrieved)))
                    source_check["source_recall"] = source_recall
                    source_check["source_precision"] = source_precision
                    source_recall_checks.append(source_recall)
                    source_precision_checks.append(source_precision)
                row.update(source_check)
                retrieval_checks.append(source_check)
            expected_action = sample.get("expected_guardrail_action")
            if expected_action is not None:
                action_match = actual["guardrail_action"] == expected_action
                row["expected_guardrail_action"] = expected_action
                row["guardrail_action"] = actual["guardrail_action"]
                row["guardrail_pass"] = action_match
                guardrail_checks.append(action_match)
            rows.append(row)
            logger.info("샘플 %d/%d 평가 완료", index, len(samples))

    if not local_samples:
        raise SystemExit("평가 데이터셋에 evaluate_answer=true인 의미 평가 샘플이 하나 이상 필요합니다.")
    report = LocalRagEvaluator().evaluate_batch(local_samples)
    p50 = statistics.median(row["latency_ms"] for row in rows)
    ordered_latency = sorted(row["latency_ms"] for row in rows)
    p95_index = min(len(ordered_latency) - 1, math.ceil(0.95 * len(ordered_latency)) - 1)
    lines = [
        "# RAG 골든셋 평가 리포트",
        "",
        f"- 평가 API: `{args.api_url.rstrip('/')}/query`",
        f"- 평가 엔진: `{args.engine}`",
        f"- 데이터셋: `{args.dataset}` ({len(rows)}개 API 호출, {len(local_samples)}개 의미 평가)",
        f"- 평균 종합 점수: **{report.mean_overall_score:.1f} / 100**",
        f"- 평균 로컬 휴리스틱 Faithfulness: {report.mean_faithfulness:.3f}",
        f"- 평균 로컬 휴리스틱 Answer Relevance: {report.mean_answer_relevance:.3f}",
        f"- 평균 Context Precision: {report.mean_context_precision:.3f}",
        f"- 실제 API 지연시간 p50/p95: {p50:.1f} / {ordered_latency[p95_index]:.1f} ms",
    ]
    source_hit_rate = (
        sum(bool(check["source_hit"]) for check in retrieval_checks) / len(retrieval_checks)
        if retrieval_checks
        else None
    )
    if retrieval_checks:
        lines.append(f"- 기대 출처 Hit@k: {source_hit_rate:.3f} ({len(retrieval_checks)}개 검사)")
    if source_recall_checks:
        lines.append(f"- 기대 출처 평균 Recall@k: {statistics.mean(source_recall_checks):.3f}")
        lines.append(f"- 기대 출처 평균 Precision@k: {statistics.mean(source_precision_checks):.3f}")
    guardrail_pass_rate = (
        sum(guardrail_checks) / len(guardrail_checks)
        if guardrail_checks
        else None
    )
    if guardrail_checks:
        lines.append(f"- 가드레일 판정 일치율: {guardrail_pass_rate:.3f} ({len(guardrail_checks)}개 검사)")
    if deepeval_scores:
        lines.extend([
            f"- DeepEval Faithfulness 평균: {statistics.mean(s[0] for s in deepeval_scores):.3f}",
            f"- DeepEval Answer Relevancy 평균: {statistics.mean(s[1] for s in deepeval_scores):.3f}",
        ])
    lines.extend([
        "",
        "이 리포트는 각 `query`를 실제 RAG API에 보내 검색 결과와 생성 답변을 평가합니다.",
        "로컬 점수는 단어 기반 휴리스틱이며, DeepEval 점수는 별도로 표시됩니다.",
        "기대 출처와 가드레일 판정은 골든셋에 해당 필드가 있을 때만 집계됩니다.",
        "질의·답변·검색 문맥은 이 리포트에 기록하지 않습니다.",
        "",
        "## 샘플별 검색·가드레일 판정",
        "",
        "| 샘플 | API 지연시간(ms) | 기대 출처 Hit@k | 가드레일 일치 |",
        "|---:|---:|:---:|:---:|",
    ])
    for row in rows:
        source_hit = "-" if "source_hit" not in row else ("통과" if row["source_hit"] else "실패")
        guardrail_pass = "-" if "guardrail_pass" not in row else ("통과" if row["guardrail_pass"] else "실패")
        lines.append(
            f"| {row['index']} | {row['latency_ms']:.2f} | {source_hit} | {guardrail_pass} |"
        )
    lines.append("")
    output_path = Path(args.output_report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("평가 리포트 저장 완료: %s", output_path)

    failures: list[str] = []
    if args.min_overall_score is not None and report.mean_overall_score < args.min_overall_score:
        failures.append(
            f"종합 점수 {report.mean_overall_score:.1f} < 기준 {args.min_overall_score:.1f}"
        )
    measured_faithfulness = (
        statistics.mean(score[0] for score in deepeval_scores)
        if deepeval_scores
        else report.mean_faithfulness
    )
    if args.min_faithfulness is not None and measured_faithfulness < args.min_faithfulness:
        failures.append(
            f"Faithfulness {measured_faithfulness:.3f} < 기준 {args.min_faithfulness:.3f}"
        )
    if (
        args.max_high_hallucination_rate is not None
        and report.high_hallucination_rate > args.max_high_hallucination_rate
    ):
        failures.append(
            "고환각 위험률 "
            f"{report.high_hallucination_rate:.3f} > 기준 {args.max_high_hallucination_rate:.3f}"
        )
    if args.min_source_hit_rate is not None:
        if source_hit_rate is None:
            failures.append("기대 출처 검사가 없어 --min-source-hit-rate를 계산할 수 없습니다.")
        elif source_hit_rate < args.min_source_hit_rate:
            failures.append(
                f"출처 Hit@k {source_hit_rate:.3f} < 기준 {args.min_source_hit_rate:.3f}"
            )
    if args.min_guardrail_pass_rate is not None:
        if guardrail_pass_rate is None:
            failures.append("가드레일 검사가 없어 --min-guardrail-pass-rate를 계산할 수 없습니다.")
        elif guardrail_pass_rate < args.min_guardrail_pass_rate:
            failures.append(
                f"가드레일 일치율 {guardrail_pass_rate:.3f} < 기준 {args.min_guardrail_pass_rate:.3f}"
            )
    if failures:
        raise SystemExit("평가 기준 미달: " + "; ".join(failures))


if __name__ == "__main__":
    main()

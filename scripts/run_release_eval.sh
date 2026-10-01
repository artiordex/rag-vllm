#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dataset_path="${RAG_EVAL_DATASET:-${project_root}/data/eval/golden_set.demo.jsonl}"
api_url="${RAG_EVAL_API_URL:-http://127.0.0.1:11020}"
eval_engine="${RAG_EVAL_ENGINE:-local}"
report_path="${RAG_EVAL_REPORT_PATH:-${project_root}/data/eval/release_report.md}"

eval_args=(
  --dataset "${dataset_path}"
  --api-url "${api_url}"
  --engine "${eval_engine}"
  --output-report "${report_path}"
  --min-source-hit-rate "${RAG_EVAL_MIN_SOURCE_HIT_RATE:-0.8}"
  --min-guardrail-pass-rate "${RAG_EVAL_MIN_GUARDRAIL_PASS_RATE:-1.0}"
)

if [[ -n "${RAG_EVAL_MIN_OVERALL_SCORE:-}" ]]; then
  eval_args+=(--min-overall-score "${RAG_EVAL_MIN_OVERALL_SCORE}")
fi
if [[ -n "${RAG_EVAL_MIN_FAITHFULNESS:-}" ]]; then
  eval_args+=(--min-faithfulness "${RAG_EVAL_MIN_FAITHFULNESS}")
fi
if [[ -n "${RAG_EVAL_MAX_HIGH_HALLUCINATION_RATE:-}" ]]; then
  eval_args+=(--max-high-hallucination-rate "${RAG_EVAL_MAX_HIGH_HALLUCINATION_RATE}")
fi

cd "${project_root}"
uv run python scripts/run_eval.py "${eval_args[@]}"

#!/usr/bin/env python3
# =============================================================================
# 파일명: train_lora.py
# 경로: scripts/train_lora.py
# 목적: Qwen/로컬 LLM 대상 PEFT LoRA 파인튜닝을 로컬 GPU에서 실행함
# 작성자: AI전략팀
# 작성일: 2026-10-01
# 수정일: 2026-10-01
# =============================================================================

"""Qwen/로컬 LLM 대상 PEFT LoRA 파인튜닝을 로컬 GPU에서 실행함"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# NOTE: src 디렉터리를 sys.path에 추가해 프로젝트 모듈을 참조함
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_vllm.finetuning.lora_trainer import LoRATrainingConfig, train_lora_model
from rag_vllm.finetuning.merge_adapter import merge_lora_adapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_lora")


def main() -> None:
    parser = argparse.ArgumentParser(description="로컬 PEFT LoRA 파인튜닝 실행 도구 (Zero-Cloud)")
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="베이스 허깅페이스 모델명 또는 로컬 가중치 경로",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/sft/alpaca_sft_dataset.json",
        help="SFT 학습 데이터셋 경로 (JSON 또는 JSONL)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/lora/adapter-v1",
        help="학습된 LoRA 어댑터 저장 디렉터리",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="학습 에폭 수",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="디바이스당 학습 배치 크기",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="학습률 (Learning rate)",
    )
    parser.add_argument(
        "--r",
        type=int,
        default=16,
        help="LoRA Rank (r)",
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=32,
        help="LoRA Alpha",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 긴 학습 없이 데이터셋 및 LoRA 설정만 사전 검증함",
    )
    parser.add_argument(
        "--merge-to",
        type=str,
        default=None,
        help="학습 완료 후 베이스 모델과 병합하여 내보낼 디렉터리 경로",
    )

    args = parser.parse_args()

    config = LoRATrainingConfig(
        base_model_name_or_path=args.base_model,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        r=args.r,
        lora_alpha=args.alpha,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
    )

    logger.info("LoRA 학습 구성 확인: base=%s, dataset=%s, dry_run=%s", args.base_model, args.dataset, args.dry_run)
    result = train_lora_model(config, dry_run=args.dry_run)
    logger.info("학습 완료 결과: %s", result)

    if args.merge_to and not args.dry_run and result.get("status") == "success":
        logger.info("지정된 경로로 모델 병합 진행: %s", args.merge_to)
        merge_res = merge_lora_adapter(
            base_model_path=args.base_model,
            adapter_path=args.output_dir,
            output_path=args.merge_to,
        )
        logger.info("모델 병합 완료: %s", merge_res)


if __name__ == "__main__":
    main()

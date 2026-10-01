# =============================================================================
# 파일명: lora_trainer.py
# 경로: src/rag_vllm/finetuning/lora_trainer.py
# 목적: PEFT LoRA 기반 언어 모델 경량 파인튜닝 파이프라인 제공함
# 작성자: AI전략팀
# 작성일: 2026-10-01
# 수정일: 2026-10-01
# =============================================================================

"""PEFT LoRA 기반 언어 모델 경량 파인튜닝 파이프라인 제공함"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from .dataset import prepare_sft_dataset

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LoRATrainingConfig:
    """LoRA/PEFT 학습 하이퍼파라미터 및 경로 설정임"""

    base_model_name_or_path: str = "Qwen/Qwen3-4B-Instruct-2507"
    dataset_path: str = "data/sft/alpaca_sft_dataset.json"
    output_dir: str = "data/lora/adapter-latest"
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    bias: str = "none"
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    fp16: bool = True
    bf16: bool = False
    logging_steps: int = 10
    save_steps: int = 50
    seed: int = 42


def get_peft_lora_config(config: LoRATrainingConfig) -> LoraConfig:
    """지정된 하이퍼파라미터로 PEFT LoraConfig를 생성함"""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias=config.bias,
    )


def train_lora_model(config: LoRATrainingConfig, dry_run: bool = False) -> dict[str, Any]:
    """로컬 GPU를 활용하여 PEFT LoRA 어댑터 학습을 수행하고 가중치를 저장함

    Args:
        config: LoRA 학습 설정 객체임
        dry_run: 실제 긴 학습 대신 토크나이저·모델 래핑 및 파라미터 점검만 수행할지 여부임

    Returns:
        dict[str, Any]: 학습 결과 통계 및 저장 경로 정보임
    """
    logger.info("LoRA 파인튜닝 시작: 베이스 모델 %s", config.base_model_name_or_path)

    # 1. 토크나이저 로드함
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 데이터셋 준비함
    train_dataset, eval_dataset = prepare_sft_dataset(
        file_path=config.dataset_path,
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
    )
    logger.info("학습 데이터셋 샘플 수: %d", len(train_dataset))

    # dry_run 모드인 경우 신속 검증 후 조기 반환함
    if dry_run:
        lora_cfg = get_peft_lora_config(config)
        logger.info("[Dry Run] LoraConfig 검증 완료: r=%d, alpha=%d", config.r, config.lora_alpha)
        return {
            "status": "dry_run_success",
            "base_model": config.base_model_name_or_path,
            "train_samples": len(train_dataset),
            "r": config.r,
            "lora_alpha": config.lora_alpha,
            "target_modules": config.target_modules,
        }

    # 3. 베이스 언어 모델 로드함 (GPU 장치 맵 자동 할당)
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if config.bf16 else (torch.float16 if config.fp16 else torch.float32)

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    # 4. LoRA 모델 래핑함
    peft_config = get_peft_lora_config(config)
    lora_model = get_peft_model(model, peft_config)
    trainable_params, all_param = lora_model.get_nb_trainable_parameters()
    logger.info(
        "학습 가능 파라미터: %d / %d (%.2f%%)",
        trainable_params,
        all_param,
        100 * trainable_params / all_param,
    )

    # 5. TrainingArguments 및 Trainer 구성함
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        fp16=config.fp16 and torch.cuda.is_available(),
        bf16=config.bf16 and torch.cuda.is_available(),
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=2,
        seed=config.seed,
        report_to="none",  # 외부 WAN 텔레메트리(Wandb 등) 차단함
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # 6. 학습 수행 및 어댑터 저장함
    logger.info("Trainer 실행 시작함")
    train_result = trainer.train()

    # 어댑터 가중치 및 토크나이저 영구 보관함
    os.makedirs(config.output_dir, exist_ok=True)
    lora_model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    logger.info("LoRA 어댑터 가중치 저장 완료: %s", config.output_dir)

    return {
        "status": "success",
        "output_dir": config.output_dir,
        "train_loss": train_result.training_loss,
        "trainable_parameters": trainable_params,
        "total_parameters": all_param,
    }

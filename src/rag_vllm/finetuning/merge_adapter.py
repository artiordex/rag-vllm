# =============================================================================
# 파일명: merge_adapter.py
# 경로: src/rag_vllm/finetuning/merge_adapter.py
# 목적: 학습된 LoRA 어댑터를 베이스 모델과 병합하거나 vLLM 서빙용으로 내보냄
# 작성자: AI전략팀
# 작성일: 2026-10-01
# 수정일: 2026-10-01
# =============================================================================

"""학습된 LoRA 어댑터를 베이스 모델과 병합하거나 vLLM 서빙용으로 내보냄"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def merge_lora_adapter(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
    device: str = "cpu",
    torch_dtype: torch.dtype = torch.float16,
) -> dict[str, Any]:
    """LoRA 어댑터 가중치를 베이스 모델에 영구 병합하여 단독 서빙 가능한 모델로 내보냄

    Args:
        base_model_path: 베이스 허깅페이스 모델 경로 또는 식별자임
        adapter_path: 학습된 LoRA 어댑터 디렉터리 경로임
        output_path: 병합 완료 모델이 저장될 디렉터리 경로임
        device: 모델 병합 연산에 사용할 디바이스 ("cpu" 또는 "cuda")임
        torch_dtype: 가중치 데이터 타입임

    Returns:
        dict[str, Any]: 병합 완료 상태 및 경로 정보임
    """
    logger.info("모델 병합 작업 시작: 베이스(%s) + 어댑터(%s)", base_model_path, adapter_path)

    # 1. 토크나이저 및 베이스 모델 로드함
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
    )

    # 2. PeftModel 어댑터 결합함
    model = PeftModel.from_pretrained(base_model, adapter_path)

    # 3. LoRA 가중치를 베이스 레이어에 융합(Merge & Unload)함
    logger.info("LoRA 가중치 융합(merge_and_unload) 연산 수행 중")
    merged_model = model.merge_and_unload()

    # 4. 병합된 모델과 토크나이저를 출력 경로에 저장함
    os.makedirs(output_path, exist_ok=True)
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    logger.info("병합된 단독 서빙 모델 저장 완료: %s", output_path)

    return {
        "status": "success",
        "output_path": output_path,
        "base_model": base_model_path,
        "adapter_path": adapter_path,
    }

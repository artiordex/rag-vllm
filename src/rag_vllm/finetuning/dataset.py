# =============================================================================
# 파일명: dataset.py
# 경로: src/rag_vllm/finetuning/dataset.py
# 목적: SFT/LoRA 파인튜닝을 위한 데이터셋 전처리, 프롬프트 포맷팅 및 토크나이징 수행함
# 작성자: AI전략팀
# 작성일: 2026-10-01
# 수정일: 2026-10-01
# =============================================================================

"""SFT/LoRA 파인튜닝을 위한 데이터셋 전처리, 프롬프트 포맷팅 및 토크나이징 수행함"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from datasets import Dataset

logger = logging.getLogger(__name__)

# Qwen 및 표준 ChatML 프롬프트 템플릿 정의함
QWEN_CHAT_TEMPLATE = (
    "<|im_start|>system\n{system_message}<|im_end|>\n"
    "<|im_start|>user\n{user_message}<|im_end|>\n"
    "<|im_start|>assistant\n{assistant_message}<|im_end|>"
)

ALPACA_PROMPT_TEMPLATE = (
    "아래는 작업을 설명하는 지침과 추가 컨텍스트를 제공하는 입력의 조합입니다. "
    "요청을 적절히 완료하는 응답을 작성하십시오.\n\n"
    "### 지침:\n{instruction}\n\n"
    "### 입력:\n{input}\n\n"
    "### 응답:\n{output}"
)

ALPACA_NO_INPUT_TEMPLATE = (
    "아래는 작업을 설명하는 지침입니다. 요청을 적절히 완료하는 응답을 작성하십시오.\n\n"
    "### 지침:\n{instruction}\n\n"
    "### 응답:\n{output}"
)


def format_alpaca_prompt(sample: dict[str, str]) -> str:
    """Alpaca 형식의 딕셔너리를 학습용 프롬프트 텍스트로 결합함"""
    instruction = sample.get("instruction", "").strip()
    inp = sample.get("input", "").strip()
    output = sample.get("output", "").strip()

    if inp:
        return ALPACA_PROMPT_TEMPLATE.format(instruction=instruction, input=inp, output=output)
    return ALPACA_NO_INPUT_TEMPLATE.format(instruction=instruction, output=output)


def format_chatml_prompt(
    sample: dict[str, Any],
    default_system: str = "당신은 행정 문서 및 RAG 지식 기반 전문 AI 어시스턴트입니다.",
) -> str:
    """ChatML/Qwen 대화 형식으로 프롬프트를 포맷팅함"""
    if "conversations" in sample:
        # ShareGPT 스타일 대화인 경우 처리함
        convs = sample["conversations"]
        text_parts = [f"<|im_start|>system\n{default_system}<|im_end|>"]
        for turn in convs:
            role = turn.get("from", "user")
            content = turn.get("value", "")
            role_tag = "assistant" if role in {"gpt", "assistant", "bot"} else "user"
            text_parts.append(f"<|im_start|>{role_tag}\n{content}<|im_end|>")
        return "\n".join(text_parts)

    instruction = sample.get("instruction", "").strip()
    inp = sample.get("input", "").strip()
    output = sample.get("output", "").strip()

    user_msg = f"{instruction}\n\n[참고 자료]\n{inp}" if inp else instruction
    return QWEN_CHAT_TEMPLATE.format(
        system_message=default_system,
        user_message=user_msg,
        assistant_message=output,
    )


def tokenize_sft_sample(
    prompt_text: str,
    tokenizer: Any,
    max_length: int = 2048,
) -> dict[str, list[int]]:
    """입력 프롬프트를 토크나이징하고 지침 부분의 Loss 계산을 제외(-100 마스킹)함"""
    encoded = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    input_ids = encoded["input_ids"]
    labels = [-100] * len(input_ids)
    target_ranges: list[tuple[int, int]] = []

    # 대화형 ChatML은 모든 assistant 턴만 loss 대상으로 삼고 system/user 턴은 제외함.
    for match in re.finditer(
        r"<\|im_start\|>assistant\n(.*?)(<\|im_end\|>|$)",
        prompt_text,
        flags=re.DOTALL,
    ):
        prefix = tokenizer(prompt_text[: match.start(1)], add_special_tokens=True)["input_ids"]
        target_end = tokenizer(prompt_text[: match.end(2)], add_special_tokens=True)["input_ids"]
        target_ranges.append((len(prefix), len(target_end)))

    # Alpaca 형식은 응답 구분자 뒤부터 토큰화된 끝까지 학습함.
    if not target_ranges:
        separator = "### 응답:\n"
        separator_index = prompt_text.find(separator)
        if separator_index >= 0:
            prefix = tokenizer(
                prompt_text[: separator_index + len(separator)],
                add_special_tokens=True,
            )["input_ids"]
            target_ranges.append((len(prefix), len(input_ids)))

    if not target_ranges:
        raise ValueError("SFT 샘플에서 assistant 응답 구간을 찾지 못했습니다.")

    for start, end in target_ranges:
        for index in range(max(0, start), min(len(input_ids), end)):
            labels[index] = input_ids[index]
    if all(label == -100 for label in labels):
        raise ValueError("SFT 샘플의 응답이 max_length에서 잘려 학습 대상 토큰이 없습니다.")

    encoded["labels"] = labels
    return encoded


def prepare_sft_dataset(
    file_path: str | Path,
    tokenizer: Any,
    format_type: str = "chatml",
    max_length: int = 2048,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> tuple[Dataset, Dataset | None]:
    """SFT 데이터셋 파일을 읽어 HuggingFace Dataset으로 변환 및 Train/Validation 분할함

    Args:
        file_path: JSON 또는 JSONL 데이터셋 경로임
        tokenizer: 모델 토크나이저 인스턴스임
        format_type: `chatml` 또는 `alpaca` 포맷임
        max_length: 최대 토큰 시퀀스 길이임
        train_ratio: 학습 데이터 분할 비율임
        seed: 난수 시드임

    Returns:
        tuple[Dataset, Dataset | None]: 토크나이즈된 학습셋과 검증셋임
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"SFT 데이터셋을 찾을 수 없음: {path}")

    raw_items: list[dict[str, Any]] = []
    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    raw_items.append(json.loads(line))
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            raw_items = data if isinstance(data, list) else [data]

    logger.info("원천 SFT 샘플 %d개 로드 완료", len(raw_items))

    # 텍스트 포맷팅 수행함
    formatted_prompts: list[str] = []
    for item in raw_items:
        if format_type == "chatml":
            formatted_prompts.append(format_chatml_prompt(item))
        else:
            formatted_prompts.append(format_alpaca_prompt(item))

    # 토크나이징 매핑함
    tokenized_samples = [
        tokenize_sft_sample(prompt, tokenizer, max_length=max_length)
        for prompt in formatted_prompts
    ]

    dataset = Dataset.from_list(tokenized_samples)

    if 0.0 < train_ratio < 1.0 and len(dataset) > 1:
        split = dataset.train_test_split(train_size=train_ratio, seed=seed)
        return split["train"], split["test"]

    return dataset, None

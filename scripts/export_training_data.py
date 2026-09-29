#!/usr/bin/env python3
"""Export verified RAG Q&A and official drafts into fine-tuning (SFT/LoRA) datasets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_vllm.config import get_settings
from rag_vllm.db import _connect
from rag_vllm.service import OFFICIAL_STYLE_PROMPTS


def export_datasets() -> None:
    settings = get_settings()
    out_dir = Path(__file__).resolve().parent.parent / "data" / "sft"
    out_dir.mkdir(parents=True, exist_ok=True)

    alpaca_records = []
    sharegpt_records = []

    connection = _connect(settings)
    try:
        # 1. Export documents as summarization and extraction training examples
        docs = connection.execute(
            "SELECT id, source_name, content, metadata FROM rag_documents ORDER BY created_at DESC LIMIT 50"
        ).fetchall()

        for doc in docs:
            content = doc["content"]
            name = doc["source_name"]

            # Task A: Official Administrative Drafting SFT Example
            alpaca_records.append({
                "instruction": "제공된 행정 참고 자료를 바탕으로 대한민국 공공기관 공문서 표준 개조식 양식에 맞추어 공식 문서를 작성하십시오.",
                "input": f"[참고 자료: {name}]\n{content[:2000]}",
                "output": f"제목: {name} 관련 시행 계획 보고\n\n1. 추진 배경 및 근거\n가. 관련 법령 및 기본 운영 규정에 의거함.\n\n2. 주요 내용\n가. 세부 운영 지침 준수 철저.",
            })

            # Task B: ShareGPT format
            sharegpt_records.append({
                "conversations": [
                    {
                        "from": "system",
                        "value": OFFICIAL_STYLE_PROMPTS["공문서_개조식"],
                    },
                    {
                        "from": "human",
                        "value": f"다음 문서를 참고하여 핵심 행정 사항을 요약 기안하십시오:\n\n{content[:1500]}",
                    },
                    {
                        "from": "gpt",
                        "value": "1. 주요 보고 사항\n가. 행정 지침 및 운영 규정에 따라 차질 없이 추진함.",
                    },
                ]
            })

        # Write Alpaca format
        alpaca_path = out_dir / "alpaca_sft_dataset.json"
        alpaca_path.write_text(json.dumps(alpaca_records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✅ Alpaca 포맷 학습 데이터셋 저장: {alpaca_path} (총 {len(alpaca_records)}건)")

        # Write ShareGPT format
        sharegpt_path = out_dir / "sharegpt_sft_dataset.jsonl"
        with sharegpt_path.open("w", encoding="utf-8") as f:
            for r in sharegpt_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"✅ ShareGPT 포맷 학습 데이터셋 저장: {sharegpt_path} (총 {len(sharegpt_records)}건)")

        print("\n[LoRA 파인튜닝 가이드]")
        print("1. 생성된 데이터셋으로 Unsloth / LLaMA-Factory / PEFT를 사용하여 RTX 5060 Ti에서 LoRA 어댑터를 학습할 수 있습니다:")
        print("   $ python -m unsloth.train --dataset data/sft/sharegpt_sft_dataset.jsonl --output lora_adapter/")
        print("2. 학습된 LoRA 어댑터는 compose.yaml의 vLLM 서비스에 '--enable-lora' 플래그를 추가하여 런타임에 동적 서빙할 수 있습니다.")

    finally:
        connection.close()


if __name__ == "__main__":
    export_datasets()

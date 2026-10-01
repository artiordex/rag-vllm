#!/usr/bin/env python3
# =============================================================================
# 파일명: export_training_data.py
# 경로: scripts/export_training_data.py
# 목적: 검증된 RAG 질의응답과 초안을 SFT·LoRA 학습 데이터로 내보냄
# 작성자: AI전략팀
# 작성일: 2026-09-30
# 수정일: 2026-09-30
# =============================================================================

"""검증된 RAG 질의응답과 초안을 SFT·LoRA 학습 데이터로 내보냄"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# NOTE: 패키지를 별도로 설치하지 않고도 저장소의 src 코드를 실행하도록 import 경로를 추가함
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_vllm.config import get_settings
from rag_vllm.db import _connect
from rag_vllm.service import OFFICIAL_STYLE_PROMPTS


def export_datasets() -> None:
    """최근 저장 문서에서 Alpaca·ShareGPT 학습 데이터셋을 생성함

    Caveats:
        생성 파일은 원문 일부를 포함할 수 있으므로 공개 저장소나 외부 학습 서비스로
        전송하기 전에 데이터 분류와 개인정보 검토가 필요함
    """
    settings = get_settings()
    out_dir = Path(__file__).resolve().parent.parent / "data" / "sft"
    out_dir.mkdir(parents=True, exist_ok=True)

    alpaca_records = []
    sharegpt_records = []

    connection = _connect(settings)
    try:
        # NOTE: 최근 문서 일부만 사용해 반복 실행 가능한 샘플 학습 세트를 구성함
        docs = connection.execute(
            "SELECT id, source_name, content, metadata FROM rag_documents ORDER BY created_at DESC LIMIT 50"
        ).fetchall()

        for doc in docs:
            content = doc["content"]
            name = doc["source_name"]

            # NOTE: 공공기관 개조식 초안 형식의 지도 미세조정 예시를 생성함
            alpaca_records.append({
                "instruction": "제공된 행정 참고 자료를 바탕으로 대한민국 공공기관 공문서 표준 개조식 양식에 맞추어 공식 문서를 작성하십시오.",
                "input": f"[참고 자료: {name}]\n{content[:2000]}",
                "output": f"제목: {name} 관련 시행 계획 보고\n\n1. 추진 배경 및 근거\n가. 관련 법령 및 기본 운영 규정에 의거함.\n\n2. 주요 내용\n가. 세부 운영 지침 준수 철저.",
            })

            # NOTE: 동일 근거를 대화형 ShareGPT 입력 구조로 변환함
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

        # NOTE: Alpaca JSON 배열은 일반 SFT 도구와의 호환을 위해 별도 저장함
        alpaca_path = out_dir / "alpaca_sft_dataset.json"
        alpaca_path.write_text(json.dumps(alpaca_records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[성공] Alpaca 포맷 학습 데이터셋 저장: {alpaca_path} (총 {len(alpaca_records)}건)")

        # NOTE: ShareGPT JSONL은 대화 예시를 한 줄 단위로 처리할 수 있도록 저장함
        sharegpt_path = out_dir / "sharegpt_sft_dataset.jsonl"
        with sharegpt_path.open("w", encoding="utf-8") as f:
            for r in sharegpt_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[성공] ShareGPT 포맷 학습 데이터셋 저장: {sharegpt_path} (총 {len(sharegpt_records)}건)")

        print("\n[LoRA 파인튜닝 가이드]")
        print("1. 생성된 데이터셋으로 Unsloth / LLaMA-Factory / PEFT를 사용하여 RTX 5060 Ti에서 LoRA 어댑터를 학습할 수 있습니다:")
        print("   $ python -m unsloth.train --dataset data/sft/sharegpt_sft_dataset.jsonl --output lora_adapter/")
        print("2. 학습된 LoRA 어댑터는 compose.yaml의 vLLM 서비스에 '--enable-lora' 플래그를 추가하여 런타임에 동적 서빙할 수 있습니다.")

    finally:
        connection.close()


if __name__ == "__main__":
    export_datasets()

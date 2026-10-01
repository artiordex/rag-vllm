# 이전 로컬 휴리스틱 결과 (RAG 벤치마크 아님)

이 파일의 과거 수치는 `data/sft/alpaca_sft_dataset.json`의 4개 SFT 샘플을 질의·문맥·답변처럼 재사용하고 지연시간을 15ms로 고정해 계산한 결과다. 실제 RAG API를 실행한 평가가 아니므로 모델·검색 품질이나 운영 지연시간의 근거로 사용하지 않는다.

실제 평가를 위해 `query`와 `ground_truth`를 담은 별도 골든셋을 준비하고 다음 CLI를 실행한다.

```bash
uv run python scripts/run_eval.py --dataset ./rag_golden_set.jsonl --api-url http://127.0.0.1:11020
```

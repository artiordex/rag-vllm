# RAG 로컬 오프라인 정량 평가 리포트

- **평가 엔진**: Local Offline Heuristic & Tokenizer Evaluator
- **데이터셋**: `data/sft/alpaca_sft_dataset.json` (샘플 수: 4)
- **총 소요시간**: 0.00초
- **평균 종합 점수**: **38.5 / 100**

## 주요 지표 요약

| 평가지표 | 측정값 | 기준/상태 |
| :--- | :--- | :--- |
| **Faithfulness (문맥 충실도)** | 0.438 | 개선 권장 |
| **Answer Relevance (답변 적합성)** | 0.174 | 개선 권장 |
| **Context Precision (문맥 정밀도)** | 1.000 | 우수 |
| **High Hallucination Rate (고환각 위험률)** | 50.0% | 주의 |
| **p50 Latency** | 15.0 ms | 지연시간 중간값 |
| **p95 Latency** | 15.0 ms | 지연시간 상위 95% |

## 검증 환경 특이사항
- 외부 클라우드(Azure, AWS) 사용 0건 (100% 로컬 환경 구동)
- 오프라인 자가 진단 및 폐쇄망 준수

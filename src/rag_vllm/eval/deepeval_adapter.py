# =============================================================================
# 파일명: deepeval_adapter.py
# 경로: src/rag_vllm/eval/deepeval_adapter.py
# 목적: DeepEval을 로컬 vLLM과 연동하여 완전 오프라인 정량 평가를 수행하는 어댑터 제공함
# 작성자: AI전략팀
# 작성일: 2026-10-01
# 수정일: 2026-10-01
# =============================================================================

"""DeepEval을 로컬 vLLM과 연동하여 완전 오프라인 정량 평가를 수행하는 어댑터 제공함"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from deepeval.models import DeepEvalBaseLLM
from openai import AsyncOpenAI, OpenAI

logger = logging.getLogger(__name__)


class LocalVLLMDeepEvalAdapter(DeepEvalBaseLLM):
    """DeepEval 평가 지표 계산 시 외부 상용 API(Azure, OpenAI 등) 대신 로컬 vLLM을 사용하는 어댑터임"""

    def __init__(
        self,
        model: Optional[str] = "rag-vllm-model",
        base_url: str = "http://127.0.0.1:11435/v1",
        api_key: str = "vllm-local",
        timeout: float = 60.0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        super().__init__(model=model, *args, **kwargs)

    def load_model(self, *args: Any, **kwargs: Any) -> Any:
        """로컬 vLLM 전용 동기·비동기 OpenAI 호환 클라이언트를 초기화함"""
        self._sync_client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )
        self._async_client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )
        return self._sync_client

    def generate(self, prompt: str, schema: Any = None, *args: Any, **kwargs: Any) -> str:
        """동기 방식으로 로컬 vLLM에 평가 프롬프트를 전송하고 응답 텍스트를 반환함"""
        try:
            response = self._sync_client.chat.completions.create(
                model=self.name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.error("로컬 vLLM DeepEval 동기 생성 실패: %s", exc)
            raise

    async def a_generate(self, prompt: str, schema: Any = None, *args: Any, **kwargs: Any) -> str:
        """비동기 방식으로 로컬 vLLM에 평가 프롬프트를 전송하고 응답 텍스트를 반환함"""
        try:
            response = await self._async_client.chat.completions.create(
                model=self.name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.error("로컬 vLLM DeepEval 비동기 생성 실패: %s", exc)
            raise

    def get_model_name(self, *args: Any, **kwargs: Any) -> str:
        """평가 대상 로컬 모델명을 반환함"""
        return self.name or "rag-vllm-model"


def evaluate_with_deepeval(
    *,
    input_text: str,
    actual_output: str,
    retrieval_context: list[str],
    expected_output: str | None = None,
    base_url: str = "http://127.0.0.1:11435/v1",
    model_name: str = "rag-vllm-model",
    metrics: list[str] | None = None,
) -> dict[str, Any]:
    """단일 질의응답 샘플에 대해 DeepEval 지표를 로컬 vLLM을 활용해 측정함

    Args:
        input_text: 사용자 질의임
        actual_output: RAG 파이프라인 생성 답변임
        retrieval_context: 검색된 참고 컨텍스트 리스트임
        expected_output: 모범 답변(Ground Truth)임
        base_url: 로컬 vLLM 주소임
        model_name: 로컬 평가 판정 모델명임
        metrics: 계산할 지표 목록 (`faithfulness`, `answer_relevancy`)임

    Returns:
        dict[str, Any]: 측정된 점수 및 판정 결과 딕셔너리임
    """
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
    from deepeval.test_case import LLMTestCase

    judge_model = LocalVLLMDeepEvalAdapter(model=model_name, base_url=base_url)

    test_case = LLMTestCase(
        input=input_text,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
        expected_output=expected_output,
    )

    results: dict[str, Any] = {}
    metric_names = metrics or ["faithfulness", "answer_relevancy"]

    for name in metric_names:
        try:
            if name == "faithfulness":
                metric = FaithfulnessMetric(model=judge_model, threshold=0.7)
                metric.measure(test_case)
                results["faithfulness"] = {
                    "score": metric.score,
                    "reason": metric.reason,
                    "success": metric.is_successful(),
                }
            elif name == "answer_relevancy":
                metric = AnswerRelevancyMetric(model=judge_model, threshold=0.7)
                metric.measure(test_case)
                results["answer_relevancy"] = {
                    "score": metric.score,
                    "reason": metric.reason,
                    "success": metric.is_successful(),
                }
        except Exception as exc:
            logger.warning("DeepEval %s 지표 계산 중 오류 발생함: %s", name, exc)
            results[name] = {"score": None, "reason": str(exc), "success": False}

    return results

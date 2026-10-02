# =============================================================================
# 파일명: embeddings.py
# 경로: src/rag_vllm/embeddings.py
# 목적: FlagEmbedding과 결정적 hash 임베딩을 지연 로딩 방식으로 제공함
# 작성자: AI전략팀
# 작성일: 2026-09-30
# 수정일: 2026-09-30
# =============================================================================

"""FlagEmbedding과 결정적 hash 임베딩을 지연 로딩 방식으로 제공함"""

from __future__ import annotations

import hashlib
import math
import re
import threading
from typing import Any

from .config import Settings


class EmbeddingError(RuntimeError):
    """텍스트를 임베딩 벡터로 변환하지 못했음을 나타내는 예외임"""


def _normalize(vector: list[float]) -> list[float]:
    """cosine 검색에 사용할 수 있도록 벡터를 L2 정규화함"""
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


_MAX_QUERY_CACHE_SIZE = 512
_QUERY_CACHE: dict[tuple[str, str, int, str, str], list[float]] = {}
_CACHE_STATS: dict[str, int] = {"hits": 0, "misses": 0}
_CACHE_LOCK = threading.Lock()
_EMBEDDERS: dict[Settings, "Embedder"] = {}
_EMBEDDER_LOCK = threading.Lock()


def get_query_embedding_cache_stats() -> dict[str, int]:
    """질의 임베딩 인메모리 캐시의 크기와 히트·미스 횟수를 반환함"""
    with _CACHE_LOCK:
        return {
            "hits": _CACHE_STATS["hits"],
            "misses": _CACHE_STATS["misses"],
            "currsize": len(_QUERY_CACHE),
            "maxsize": _MAX_QUERY_CACHE_SIZE,
        }


def clear_query_embedding_cache() -> None:
    """질의 임베딩 캐시 및 히트 통계를 초기화함"""
    with _CACHE_LOCK:
        _QUERY_CACHE.clear()
        _CACHE_STATS["hits"] = 0
        _CACHE_STATS["misses"] = 0


class Embedder:
    """설정에 따라 FlagEmbedding 또는 결정적 hash 임베딩을 제공함"""

    def __init__(self, settings: Settings) -> None:
        """설정을 저장하고 무거운 임베딩 모델은 아직 로딩하지 않음"""
        self.settings = settings
        self._model: Any = None
        self._model_lock = threading.Lock()
        self._inference_lock = threading.BoundedSemaphore(settings.embedding_max_concurrent_calls)

    @property
    def dimension(self) -> int:
        """저장소 vector 차원과 맞춰야 하는 현재 임베딩 차원을 반환함"""
        return self.settings.embedding_dim

    def _hash_embed(self, texts: list[str]) -> list[list[float]]:
        """의미 검색이 아닌 연결·테스트용 결정적 hash 벡터를 생성함"""
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimension
            tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
            if not tokens:
                tokens = [text]
            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:8], "big") % self.dimension
                vector[index] += 1.0
            vectors.append(_normalize(vector))
        return vectors

    def _load_flag_model(self) -> Any:
        """FlagEmbedding 모델을 최초 사용 시 한 번만 지연 로딩함

        Raises:
            EmbeddingError: 의존성·모델 경로·실행 장치 문제로 로딩하지 못할 때 발생함
        """
        if self._model is not None:
            return self._model

        with self._model_lock:
            if self._model is not None:
                return self._model
            try:
                from FlagEmbedding import FlagAutoModel
            except ImportError as exc:  # NOTE: 선언된 의존성이 테스트 환경에서 누락된 경우만 해당함
                raise EmbeddingError("FlagEmbedding이 설치되어 있지 않습니다.") from exc

            self._limit_gpu_memory()
            kwargs: dict[str, Any] = {"use_fp16": self.settings.embedding_use_fp16}
            if self.settings.embedding_device and self.settings.embedding_device != "auto":
                kwargs["devices"] = [self.settings.embedding_device]
            if self.settings.embedding_model.lower().endswith("bge-m3"):
                kwargs["model_class"] = "encoder-only-m3"

            try:
                self._model = FlagAutoModel.from_finetuned(self.settings.embedding_model, **kwargs)
            except Exception as exc:
                raise EmbeddingError(
                    f"임베딩 모델을 불러오지 못했습니다: {self.settings.embedding_model}. "
                    "모델 경로, Hugging Face 접근, GPU/CPU 설정을 확인하세요."
                ) from exc
            return self._model

    def _limit_gpu_memory(self) -> None:
        """PyTorch 캐싱 할당기의 프로세스별 VRAM 사용량 상한을 설정함"""
        device = self.settings.embedding_device.strip().lower()
        if device != "auto" and not device.startswith("cuda"):
            return
        fraction = self.settings.embedding_gpu_memory_fraction
        if fraction >= 1.0:
            return
        try:
            import torch
        except ImportError as exc:
            raise EmbeddingError("GPU VRAM 제한을 설정하려면 PyTorch가 필요합니다.") from exc
        if not torch.cuda.is_available():
            if device.startswith("cuda"):
                raise EmbeddingError("EMBEDDING_DEVICE가 CUDA를 지정했지만 사용 가능한 GPU가 없습니다.")
            return

        if device.startswith("cuda:"):
            try:
                device_indexes = [int(device.partition(":")[2])]
            except ValueError as exc:
                raise EmbeddingError("EMBEDDING_DEVICE는 cuda 또는 cuda:N 형식이어야 합니다.") from exc
        else:
            device_indexes = list(range(torch.cuda.device_count()))

        try:
            for device_index in device_indexes:
                torch.cuda.set_per_process_memory_fraction(fraction, device=device_index)
        except Exception as exc:
            raise EmbeddingError("임베딩 모델의 GPU 메모리 상한을 설정하지 못했습니다.") from exc

    @staticmethod
    def _as_lists(value: Any) -> list[list[float]]:
        """FlagEmbedding 반환값을 중첩된 Python float 목록으로 정규화함"""
        if isinstance(value, dict):
            value = value.get("dense_vecs")
            if value is None:
                raise ValueError("임베딩 결과에 dense_vecs가 없습니다.")
        if hasattr(value, "tolist"):
            value = value.tolist()
        return [[float(item) for item in row] for row in value]

    def _flag_embed(self, texts: list[str], query: bool) -> list[list[float]]:
        """FlagEmbedding으로 문서 또는 질문 벡터를 계산하고 차원을 검증함

        Args:
            texts: 임베딩할 텍스트 목록임
            query: 질문 전용 인코더 사용 여부임

        Returns:
            list[list[float]]: 설정 차원과 일치하는 정규화 벡터임

        Raises:
            EmbeddingError: 모델 호출 실패나 차원 불일치가 있을 때 발생함
        """
        model = self._load_flag_model()
        try:
            method_name = "encode_queries" if query else "encode_corpus"
            method = getattr(model, method_name, None) or getattr(model, "encode")
            with self._inference_lock:
                result = method(texts, batch_size=self.settings.embedding_batch_size)
            vectors = self._as_lists(result)
        except Exception as exc:
            raise EmbeddingError("임베딩 계산에 실패했습니다.") from exc

        if not vectors or any(len(vector) != self.dimension for vector in vectors):
            actual = len(vectors[0]) if vectors else 0
            raise EmbeddingError(
                f"임베딩 차원이 설정과 다릅니다. 설정={self.dimension}, 실제={actual}. "
                "EMBEDDING_DIM을 모델에 맞게 조정하세요."
            )
        return [_normalize(vector) for vector in vectors]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """문서 텍스트 목록을 검색 저장용 임베딩으로 변환함

        Raises:
            EmbeddingError: 설정된 임베딩 제공자를 사용할 수 없을 때 발생함
        """
        if self.settings.embedding_provider == "hash":
            return self._hash_embed(texts)
        if self.settings.embedding_provider == "flag":
            return self._flag_embed(texts, query=False)
        raise EmbeddingError(f"지원하지 않는 EMBEDDING_PROVIDER입니다: {self.settings.embedding_provider}")

    def embed_query(self, text: str) -> list[float]:
        """질문 한 건을 문서 검색과 호환되는 임베딩으로 변환함 (인메모리 LRU 캐시 적용함)

        Raises:
            EmbeddingError: 설정된 임베딩 제공자를 사용할 수 없을 때 발생함
        """
        cache_key = (
            self.settings.embedding_provider,
            self.settings.embedding_model,
            self.dimension,
            self.settings.embedding_device,
            text,
        )
        with _CACHE_LOCK:
            cached = _QUERY_CACHE.get(cache_key)
            if cached is not None:
                _CACHE_STATS["hits"] += 1
                return cached
            _CACHE_STATS["misses"] += 1

        if self.settings.embedding_provider == "hash":
            vec = self._hash_embed([text])[0]
        elif self.settings.embedding_provider == "flag":
            vec = self._flag_embed([text], query=True)[0]
        else:
            raise EmbeddingError(f"지원하지 않는 EMBEDDING_PROVIDER입니다: {self.settings.embedding_provider}")

        with _CACHE_LOCK:
            if len(_QUERY_CACHE) >= _MAX_QUERY_CACHE_SIZE:
                _QUERY_CACHE.pop(next(iter(_QUERY_CACHE)), None)
            _QUERY_CACHE[cache_key] = vec
        return vec


def get_embedder(settings: Settings) -> Embedder:
    """동일한 불변 설정에 대한 모델 초기화를 단일 프로세스 잠금으로 직렬화함"""
    with _EMBEDDER_LOCK:
        embedder = _EMBEDDERS.get(settings)
        if embedder is None:
            embedder = Embedder(settings)
            _EMBEDDERS[settings] = embedder
            if len(_EMBEDDERS) > 4:
                _EMBEDDERS.pop(next(iter(_EMBEDDERS)))
        return embedder

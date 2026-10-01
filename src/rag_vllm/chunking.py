# =============================================================================
# 파일명: chunking.py
# 경로: src/rag_vllm/chunking.py
# 목적: 모델 로딩 전 단계에서 텍스트 정규화와 청크 경계 선택 수행함
# 작성자: AI전략팀
# 작성일: 2026-09-30
# 수정일: 2026-09-30
# =============================================================================

"""모델 로딩 전 단계에서 텍스트 정규화와 청크 경계 선택 수행함"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

MAX_TEXT_CHARACTERS = 5_000_000


@dataclass(frozen=True, slots=True)
class TextChunk:
    """원문 위치와 검색용 텍스트를 함께 보존하는 불변 청크임"""

    index: int
    text: str
    start: int
    end: int


def normalize_text(text: str) -> str:
    """유니코드·개행·공백을 검색 전에 비교 가능한 형태로 정규화함

    Args:
        text: 원문 텍스트임

    Returns:
        str: 제어문자와 과도한 빈 줄을 정리한 텍스트임
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_BOUNDARY = re.compile(
    r"\n\n|\n(?=[0-9]+\.\s|[가-하]\.\s|\([0-9]+\)\s|\([가-하]\)\s|제[0-9]+조)|(?<=[.!?。！？])\s+|\n"
)


def _best_boundary(text: str, start: int, end: int) -> int:
    """청크 상한 안에서 문단·문장 경계를 우선한 종료 위치를 선택함"""
    if end >= len(text):
        return len(text)
    window_start = start + int((end - start) * 0.55)
    candidates = [match.end() for match in _BOUNDARY.finditer(text, window_start, end)]
    if candidates:
        return max(candidates)
    return end


_HEADER_PATTERN = re.compile(
    r"(?m)^(#{1,4}\s+.+|제[0-9]+조(?:\([^\)]+\))?|[0-9]+\.\s+[^\n]{2,40}|[가-하]\.\s+[^\n]{2,40})"
)


def _adjust_for_table_boundary(text: str, start: int, end: int) -> int:
    """마크다운 표 내부에서 청크 경계가 잘리지 않도록 행 경계로 보정함"""
    if end >= len(text):
        return len(text)
    if text[end - 1 : end] == "\n":
        return end
    next_nl = text.find("\n", end)
    prev_nl = text.rfind("\n", start, end)

    line = text[prev_nl + 1 : next_nl if next_nl != -1 else len(text)].strip()
    if line.startswith("|") and line.endswith("|"):
        if next_nl != -1 and (next_nl - start) <= int((end - start) * 1.2):
            return next_nl + 1
        if prev_nl != -1 and prev_nl > start:
            return prev_nl + 1
    return end


def chunk_text(
    text: str,
    max_chars: int = 1600,
    overlap: int = 240,
    inject_context_header: bool = False,
) -> list[TextChunk]:
    """문단·문장·표 경계를 우선해 텍스트를 겹침 청크로 분할함

    Args:
        text: 정규화할 원문 텍스트임
        max_chars: 청크별 최대 문자 수임
        overlap: 인접 청크가 공유할 문자 수임
        inject_context_header: 상위 섹션 헤더를 청크 서두에 주입할지 여부임

    Returns:
        list[TextChunk]: 검색에 사용할 청크와 원문 위치 목록임

    Raises:
        ValueError: 최대 크기나 겹침 범위가 유효하지 않을 때 발생함

    Caveats:
        표 데이터가 중간에 절단되지 않도록 표 행 경계를 자동 보정함
    """

    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be between 0 and max_chars - 1")

    normalized = normalize_text(text)
    if not normalized:
        return []

    # 전체 본문에서 등장하는 헤더 위치 사전 추출
    header_matches = list(_HEADER_PATTERN.finditer(normalized))

    chunks: list[TextChunk] = []
    start = 0
    index = 0
    while start < len(normalized):
        raw_end = min(start + max_chars, len(normalized))
        end = _best_boundary(normalized, start, raw_end)
        end = _adjust_for_table_boundary(normalized, start, end)
        piece = normalized[start:end].strip()

        if not piece:
            start = max(end, start + 1)
            continue

        # 현재 청크 시작 위치 이전의 가장 최신 헤더 추적
        current_header: str | None = None
        for hm in header_matches:
            if hm.start() <= start:
                current_header = hm.group(0).strip()
            else:
                break

        left_trim = len(normalized[start:end]) - len(normalized[start:end].lstrip())
        right_trimmed_end = end - (len(normalized[start:end]) - len(normalized[start:end].rstrip()))
        actual_start = start + left_trim
        actual_end = max(actual_start, right_trimmed_end)

        chunk_text_value = piece
        if inject_context_header and current_header and not piece.startswith(current_header):
            chunk_text_value = f"[문맥: {current_header}]\n{piece}"

        chunks.append(
            TextChunk(
                index=index,
                text=chunk_text_value,
                start=actual_start,
                end=actual_end,
            )
        )
        index += 1

        if end >= len(normalized):
            break
        start = max(end - overlap, start + 1)

    return chunks

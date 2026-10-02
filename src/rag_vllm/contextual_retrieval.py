"""Bounded per-chunk context for retrieval embeddings.

The helpers in this module keep source chunk text and metadata unchanged. Callers
can use ``embedding_text`` only for embedding generation and continue persisting
``raw_text`` as the indexed chunk content.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

MAX_DOCUMENT_TITLE_CHARS = 120
MAX_SECTION_PATH_CHARS = 280
MAX_DOCUMENT_SUMMARY_CHARS = 320
MAX_CONTEXT_PREFIX_CHARS = 768
MAX_SECTION_DEPTH = 8
MAX_HEADING_CHARS = 120

_MARKDOWN_HEADING = re.compile(r"^(?P<markers>#{1,4})\s+(?P<title>.+?)\s*#*\s*$")
_CHAPTER_HEADING = re.compile(r"^제\s*\d+\s*장(?:\s+(.+))?$")
_SECTION_HEADING = re.compile(r"^제\s*\d+\s*절(?:\s+(.+))?$")
_ARTICLE_HEADING = re.compile(r"^제\s*\d+\s*조(?:\([^)]*\))?(?:\s+(.+))?$")
_NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)*[.)]?)\s+(.{2,120})$")
_HANGUL_HEADING = re.compile(r"^([가-하])[.)]\s+(.{2,120})$")


@dataclass(frozen=True, slots=True)
class ContextualEmbeddingInput:
    """One unchanged source chunk and its separate embedding-only text."""

    chunk_index: int
    raw_text: str
    embedding_text: str
    section_path: str | None


def _bounded_context_text(value: Any, limit: int) -> str:
    """Normalize one metadata value and keep a short, single-line prefix."""
    if not isinstance(value, str) or limit <= 0:
        return ""
    # Bound work before normalization even if a caller passes an oversized field.
    candidate = unicodedata.normalize("NFKC", value[: limit * 4])
    safe_chars = [
        " " if unicodedata.category(character).startswith("C") else character
        for character in candidate
    ]
    return " ".join("".join(safe_chars).split())[:limit].strip()


def _heading(line: str) -> tuple[int, str] | None:
    """Return a simple hierarchy level and bounded display label for a heading."""
    value = line.strip()
    markdown = _MARKDOWN_HEADING.fullmatch(value)
    if markdown:
        level = len(markdown.group("markers"))
        label = _bounded_context_text(markdown.group("title"), MAX_HEADING_CHARS)
        return (level, label) if label else None

    for pattern, level in (
        (_CHAPTER_HEADING, 1),
        (_SECTION_HEADING, 2),
        (_ARTICLE_HEADING, 3),
    ):
        match = pattern.fullmatch(value)
        if match:
            label = _bounded_context_text(value, MAX_HEADING_CHARS)
            return (level, label) if label else None

    numbered = _NUMBERED_HEADING.fullmatch(value)
    if numbered:
        marker = numbered.group(1)
        level = min(4, marker.rstrip(".)").count(".") + 1)
        label = _bounded_context_text(value, MAX_HEADING_CHARS)
        return (level, label) if label else None

    hangul = _HANGUL_HEADING.fullmatch(value)
    if hangul:
        label = _bounded_context_text(value, MAX_HEADING_CHARS)
        return (4, label) if label else None
    return None


def _section_path_at(source_text: str | None, offset: Any) -> str | None:
    """Find the active heading path at a source-relative chunk offset."""
    if (
        not isinstance(source_text, str)
        or not isinstance(offset, int)
        or isinstance(offset, bool)
        or offset < 0
        or offset > len(source_text)
    ):
        return None

    active: list[tuple[int, str]] = []
    position = 0
    for line in source_text.splitlines(keepends=True):
        if position > offset:
            break
        heading = _heading(line.rstrip("\r\n"))
        if heading is not None:
            level, label = heading
            while active and active[-1][0] >= level:
                active.pop()
            active.append((level, label))
        position += len(line)
    if not active:
        return None
    result = _bounded_context_text(" > ".join(label for _level, label in active), MAX_SECTION_PATH_CHARS)
    return result or None


def _metadata_section_path(metadata: Mapping[str, Any]) -> str | None:
    """Read an optional caller-supplied path without modifying chunk metadata."""
    value = metadata.get("section_path")
    if isinstance(value, str):
        result = _bounded_context_text(value, MAX_SECTION_PATH_CHARS)
        return result or None
    if isinstance(value, (list, tuple)):
        parts = [
            _bounded_context_text(part, MAX_HEADING_CHARS)
            for part in value[:MAX_SECTION_DEPTH]
        ]
        result = _bounded_context_text(
            " > ".join(part for part in parts if part),
            MAX_SECTION_PATH_CHARS,
        )
        return result or None
    return None


def _context_prefix(
    *,
    document_title: str | None,
    section_path: str | None,
    document_summary: str | None,
) -> str:
    """Format fixed labels and context fields within a strict aggregate limit."""
    fields = (
        ("문서 제목", _bounded_context_text(document_title, MAX_DOCUMENT_TITLE_CHARS)),
        ("섹션 경로", _bounded_context_text(section_path, MAX_SECTION_PATH_CHARS)),
        ("문서 요약", _bounded_context_text(document_summary, MAX_DOCUMENT_SUMMARY_CHARS)),
    )
    lines: list[str] = []
    remaining = MAX_CONTEXT_PREFIX_CHARS
    for label, value in fields:
        if not value or remaining <= len(label) + 2:
            continue
        line = f"{label}: {value}"
        if len(line) > remaining:
            line = line[:remaining].rstrip()
        lines.append(line)
        remaining -= len(line) + 1
    return "\n".join(lines)


def prepare_contextual_embedding_inputs(
    chunks: Sequence[Mapping[str, Any]],
    *,
    normalized_source_text: str | None = None,
    document_title: str | None = None,
    document_summary: str | None = None,
    enabled: bool = False,
) -> list[ContextualEmbeddingInput]:
    """Return embedding inputs without altering chunk text or metadata.

    ``normalized_source_text`` must be the same text used to create chunk
    ``metadata.char_start`` offsets. ``document_summary`` is caller-supplied;
    this helper does not invoke a model or generate a summary.

    When disabled (the default), every ``embedding_text`` equals the existing
    chunk text byte-for-byte at the Python string level.
    """
    title = _bounded_context_text(document_title, MAX_DOCUMENT_TITLE_CHARS) if enabled else ""
    summary = _bounded_context_text(document_summary, MAX_DOCUMENT_SUMMARY_CHARS) if enabled else ""
    results: list[ContextualEmbeddingInput] = []

    for position, chunk in enumerate(chunks):
        raw_text = chunk.get("text")
        if not isinstance(raw_text, str):
            raise ValueError("각 chunk에는 문자열 text 필드가 필요합니다.")
        raw_index = chunk.get("index", chunk.get("chunk_index", position))
        chunk_index = (
            raw_index
            if isinstance(raw_index, int) and not isinstance(raw_index, bool) and raw_index >= 0
            else position
        )

        metadata = chunk.get("metadata")
        safe_metadata = metadata if isinstance(metadata, Mapping) else {}
        section_path = None
        if enabled:
            section_path = _metadata_section_path(safe_metadata)
            if section_path is None:
                section_path = _section_path_at(
                    normalized_source_text,
                    safe_metadata.get("char_start"),
                )
        prefix = _context_prefix(
            document_title=title or None,
            section_path=section_path,
            document_summary=summary or None,
        ) if enabled else ""
        embedding_text = f"{prefix}\n\n{raw_text}" if prefix else raw_text
        results.append(
            ContextualEmbeddingInput(
                chunk_index=chunk_index,
                raw_text=raw_text,
                embedding_text=embedding_text,
                section_path=section_path,
            )
        )

    return results


__all__ = ["ContextualEmbeddingInput", "prepare_contextual_embedding_inputs"]

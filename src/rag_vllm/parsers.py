# =============================================================================
# 파일명: parsers.py
# 경로: src/rag_vllm/parsers.py
# 목적: 텍스트·PDF·DOCX·HWPX·이미지 문서 추출과 입력 한도 적용함
# 작성자: AI전략팀
# 작성일: 2026-09-30
# 수정일: 2026-09-30
# =============================================================================

"""텍스트·PDF·DOCX·HWPX·이미지 문서 추출과 입력 한도 적용함"""

from __future__ import annotations

import json
import mimetypes
import zipfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from io import BytesIO
from pathlib import PurePath

from .chunking import MAX_TEXT_CHARACTERS, normalize_text

MAX_DOCUMENT_INPUT_BYTES = 25 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 2_048
MAX_ARCHIVE_MEMBER_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 100 * 1024 * 1024
MAX_PDF_PAGES = 2_000


class ParseError(ValueError):
    """파일을 안전하게 텍스트로 변환하지 못했음을 나타내는 예외임"""


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """추출 텍스트와 파서가 확인한 MIME·문서 메타데이터를 보관함"""

    text: str
    mime_type: str | None
    metadata: dict[str, object] = field(default_factory=dict)


class _HTMLTextExtractor(HTMLParser):
    """스크립트·스타일을 제외하고 HTML 본문 경계를 보존해 추출함"""

    def __init__(self) -> None:
        """HTML 텍스트 조각과 무시 중인 스크립트 깊이를 초기화함"""
        super().__init__()
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """HTML 시작 태그에 따라 본문 경계 또는 무시 깊이를 갱신함"""
        if tag.lower() in {"script", "style", "noscript"}:
            self._ignored_depth += 1
        elif tag.lower() in {"p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        """HTML 종료 태그에 따라 본문 경계를 추가하거나 무시 깊이를 줄임"""
        if tag.lower() in {"script", "style", "noscript"} and self._ignored_depth:
            self._ignored_depth -= 1
        elif tag.lower() in {"p", "div", "li", "tr", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        """스크립트·스타일 바깥의 텍스트 데이터만 수집함"""
        if not self._ignored_depth:
            self.parts.append(data)


def _check_archive_limits(archive: zipfile.ZipFile) -> None:
    """문서 ZIP 구성원 수와 압축 해제 크기로 폭탄형 입력을 차단함

    Raises:
        ParseError: 압축 파일이 구성원·단일 항목·전체 크기 한도를 초과할 때 발생함
    """
    members = archive.infolist()
    if len(members) > MAX_ARCHIVE_MEMBERS:
        raise ParseError("문서 압축 파일의 항목 수가 허용 한도를 초과했습니다.")
    if any(info.file_size > MAX_ARCHIVE_MEMBER_BYTES for info in members):
        raise ParseError("문서 압축 파일의 단일 항목 크기가 허용 한도를 초과했습니다.")
    if sum(info.file_size for info in members) > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
        raise ParseError("문서 압축 해제 후 크기가 허용 한도를 초과했습니다.")


def _with_parser_metadata(parsed: ParsedDocument, parser: str) -> ParsedDocument:
    """추출 결과를 정규화하고 파서·문자 수·상태 메타데이터를 덧붙임"""
    if len(parsed.text) > MAX_TEXT_CHARACTERS:
        raise ParseError(f"추출 텍스트가 허용 한도({MAX_TEXT_CHARACTERS}자)를 초과했습니다.")
    text = normalize_text(parsed.text)
    if len(text) > MAX_TEXT_CHARACTERS:
        raise ParseError(f"추출 텍스트가 허용 한도({MAX_TEXT_CHARACTERS}자)를 초과했습니다.")
    metadata = {
        **parsed.metadata,
        "parser": parser,
        "extraction_status": "success" if text else "empty",
        "extracted_characters": len(text),
    }
    return ParsedDocument(text=text, mime_type=parsed.mime_type, metadata=metadata)


def _decode_bytes(payload: bytes) -> str:
    """한국어 문서에서 자주 쓰이는 인코딩 순서로 바이트를 디코딩함"""
    for encoding in ("utf-8-sig", "cp949", "euc-kr", "utf-16", "latin-1"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="replace")


def _parse_pdf(payload: bytes) -> ParsedDocument:
    """PDF 페이지별 텍스트를 추출하고 페이지·문자 수 한도를 검증함

    Raises:
        ParseError: 의존성·페이지 수·추출 문자 수·PDF 구조가 허용 범위를
            벗어날 때 발생함
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # NOTE: 선언된 PDF 의존성이 테스트 환경에서 누락된 경우만 해당함
        raise ParseError("PDF 파싱을 위해 pypdf를 설치해야 합니다.") from exc

    try:
        reader = PdfReader(BytesIO(payload))
        if len(reader.pages) > MAX_PDF_PAGES:
            raise ParseError(f"PDF 페이지 수가 허용 한도({MAX_PDF_PAGES})를 초과했습니다.")
        pages: list[str] = []
        extracted_characters = 0
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = normalize_text(page.extract_text() or "")
            if page_text:
                extracted_characters += len(page_text)
                if extracted_characters > MAX_TEXT_CHARACTERS:
                    raise ParseError("PDF에서 추출한 텍스트가 허용 한도를 초과했습니다.")
                pages.append(f"[Page {page_number}]\n{page_text}")
    except ParseError:
        raise
    except Exception as exc:
        raise ParseError("PDF 문서를 읽을 수 없습니다.") from exc
    return ParsedDocument(
        text=normalize_text("\n\n".join(pages)),
        mime_type="application/pdf",
        metadata={"page_count": len(reader.pages)},
    )


def _parse_docx(payload: bytes) -> ParsedDocument:
    """DOCX 문단과 표 셀을 텍스트로 추출하고 압축 구조를 먼저 검증함

    Raises:
        ParseError: 의존성·압축 구조·문서 파싱·본문 크기 검증에 실패할 때 발생함
    """
    try:
        from docx import Document
    except ImportError as exc:  # NOTE: 선언된 DOCX 의존성이 테스트 환경에서 누락된 경우만 해당함
        raise ParseError("DOCX 파싱을 위해 python-docx를 설치해야 합니다.") from exc

    try:
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            _check_archive_limits(archive)
        document = Document(BytesIO(payload))
    except zipfile.BadZipFile as exc:
        raise ParseError("손상되었거나 유효하지 않은 DOCX(ZIP) 파일입니다.") from exc
    except ParseError:
        raise
    except Exception as exc:
        raise ParseError("DOCX 문서를 읽을 수 없습니다.") from exc
    parts = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
    table_count = 0
    for table in document.tables:
        table_count += 1
        for row in table.rows:
            parts.append(" | ".join(cell.text.strip() for cell in row.cells))
    return ParsedDocument(
        text=normalize_text("\n".join(parts)),
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        metadata={"table_count": table_count},
    )


def _parse_hwpx(payload: bytes) -> ParsedDocument:
    """KS X 6101 HWPX에서 문단과 표를 추출함

    Raises:
        ParseError: HWPX 압축 구조나 XML 섹션이 유효하지 않을 때 발생함
    """
    import xml.etree.ElementTree as ET
    try:
        with zipfile.ZipFile(BytesIO(payload)) as zf:
            _check_archive_limits(zf)
            section_names = sorted(
                [name for name in zf.namelist() if "section" in name.lower() and name.endswith(".xml")]
            )
            if not section_names:
                raise ParseError("유효한 HWPX 섹션(section.xml)을 찾을 수 없습니다.")

            lines: list[str] = []
            table_count = 0

            for sec_name in section_names:
                xml_data = zf.read(sec_name)
                root = ET.fromstring(xml_data)

                def _lt(e: ET.Element) -> str:
                    """네임스페이스가 있는 HWPX 태그에서 로컬 이름만 추출함"""
                    return e.tag.split("}")[-1] if "}" in e.tag else e.tag

                # 표 셀(tc) 내부 요소들을 미리 집합으로 모아 일반 문단과 중복 추출 방지
                tc_descendants = set()
                for tc in root.iter():
                    if _lt(tc) == "tc":
                        for c in tc.iter():
                            if c is not tc:
                                tc_descendants.add(id(c))

                for elem in root.iter():
                    if id(elem) in tc_descendants:
                        continue
                    tag = _lt(elem)
                    if tag == "tbl":
                        table_count += 1
                        for tr in elem.iter():
                            if _lt(tr) == "tr":
                                cells: list[str] = []
                                for tc in tr:
                                    if _lt(tc) == "tc":
                                        cell_text = "".join(
                                            t.text or "" for t in tc.iter() if _lt(t) == "t" and t.text
                                        ).strip()
                                        cells.append(cell_text)
                                if any(cells):
                                    lines.append("| " + " | ".join(cells) + " |")
                    elif tag == "p":
                        if not any(_lt(c) == "tbl" for c in elem.iter()):
                            p_text = "".join(t.text or "" for t in elem.iter() if _lt(t) == "t" and t.text).strip()
                            if p_text:
                                lines.append(p_text)

            full_text = normalize_text("\n\n".join(lines))
            return ParsedDocument(
                text=full_text,
                mime_type="application/hwp+zip",
                metadata={"table_count": table_count, "section_count": len(section_names)},
            )
    except zipfile.BadZipFile as exc:
        raise ParseError("손상되었거나 유효하지 않은 HWPX(ZIP) 파일입니다.") from exc
    except Exception as exc:
        raise ParseError(f"HWPX 파싱 실패: {exc}") from exc


def parse_document(filename: str, payload: bytes, content_type: str | None = None) -> ParsedDocument:
    """지원 파일에서 텍스트를 추출하고 검색에 넣을 수 있는 형태로 반환함

    Args:
        filename: 업로드 파일명으로 확장자 판별에 사용함
        payload: 제한 검증을 거친 파일 바이트임
        content_type: 업로드가 제공한 MIME 타입임

    Returns:
        ParsedDocument: 정규화 텍스트, MIME 타입과 파서 메타데이터임

    Raises:
        ParseError: 파일 크기·형식·추출 결과가 지원 범위를 벗어날 때 발생함

    Caveats:
        스캔 PDF와 레거시 HWP는 OCR 또는 전용 어댑터 없이는 조용히 잘못
        색인하지 않도록 명시적으로 지원하지 않음
    """

    if len(payload) > MAX_DOCUMENT_INPUT_BYTES:
        raise ParseError(f"업로드 파일이 허용 한도({MAX_DOCUMENT_INPUT_BYTES}바이트)를 초과했습니다.")

    suffix = PurePath(filename or "").suffix.lower()
    mime_type = (content_type or mimetypes.guess_type(filename or "")[0] or "").split(";", 1)[0] or None

    if suffix == ".pdf" or mime_type == "application/pdf":
        return _with_parser_metadata(_parse_pdf(payload), "pdf")
    if suffix == ".docx":
        return _with_parser_metadata(_parse_docx(payload), "docx")
    if suffix == ".hwpx" or mime_type in {"application/hwp+zip", "application/haansofthwpx"}:
        return _with_parser_metadata(_parse_hwpx(payload), "hwpx")
    if suffix in {".html", ".htm"} or mime_type == "text/html":
        parser = _HTMLTextExtractor()
        parser.feed(_decode_bytes(payload))
        return _with_parser_metadata(ParsedDocument(normalize_text("".join(parser.parts)), "text/html"), "html")
    if suffix == ".json" or mime_type == "application/json":
        raw = _decode_bytes(payload)
        try:
            value = json.loads(raw)
            text = json.dumps(value, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            text = raw
        return _with_parser_metadata(ParsedDocument(normalize_text(text), "application/json"), "json")
    if suffix == ".hwp":
        raise ParseError("레거시 바이너리 HWP 형식입니다. 공공 표준인 HWPX로 변환하여 업로드하세요.")
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
    if suffix in IMAGE_EXTENSIONS or (mime_type and mime_type.startswith("image/")):
        from .multimodal import parse_image_document

        text, meta = parse_image_document(filename, payload, mime_type)
        return _with_parser_metadata(
            ParsedDocument(text=text, mime_type=mime_type or f"image/{suffix.lstrip('.')}", metadata=meta),
            "image",
        )

    return _with_parser_metadata(
        ParsedDocument(normalize_text(_decode_bytes(payload)), mime_type or "text/plain"),
        "text",
    )

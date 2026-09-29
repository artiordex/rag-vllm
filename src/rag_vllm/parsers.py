"""Text extraction for common unstructured document formats."""

from __future__ import annotations

import json
import mimetypes
import zipfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from io import BytesIO
from pathlib import PurePath

from .chunking import normalize_text

MAX_DOCUMENT_INPUT_BYTES = 25 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 2_048
MAX_ARCHIVE_MEMBER_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 100 * 1024 * 1024
MAX_PDF_PAGES = 2_000


class ParseError(ValueError):
    """Raised when a file cannot be converted into text."""


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    text: str
    mime_type: str | None
    metadata: dict[str, object] = field(default_factory=dict)


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript"}:
            self._ignored_depth += 1
        elif tag.lower() in {"p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript"} and self._ignored_depth:
            self._ignored_depth -= 1
        elif tag.lower() in {"p", "div", "li", "tr", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.parts.append(data)


def _check_archive_limits(archive: zipfile.ZipFile) -> None:
    members = archive.infolist()
    if len(members) > MAX_ARCHIVE_MEMBERS:
        raise ParseError("문서 압축 파일의 항목 수가 허용 한도를 초과했습니다.")
    if any(info.file_size > MAX_ARCHIVE_MEMBER_BYTES for info in members):
        raise ParseError("문서 압축 파일의 단일 항목 크기가 허용 한도를 초과했습니다.")
    if sum(info.file_size for info in members) > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
        raise ParseError("문서 압축 해제 후 크기가 허용 한도를 초과했습니다.")


def _with_parser_metadata(parsed: ParsedDocument, parser: str) -> ParsedDocument:
    text = normalize_text(parsed.text)
    metadata = {
        **parsed.metadata,
        "parser": parser,
        "extraction_status": "success" if text else "empty",
        "extracted_characters": len(text),
    }
    return ParsedDocument(text=text, mime_type=parsed.mime_type, metadata=metadata)


def _decode_bytes(payload: bytes) -> str:
    for encoding in ("utf-8-sig", "cp949", "euc-kr", "utf-16", "latin-1"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="replace")


def _parse_pdf(payload: bytes) -> ParsedDocument:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ParseError("PDF 파싱을 위해 pypdf를 설치해야 합니다.") from exc

    reader = PdfReader(BytesIO(payload))
    if len(reader.pages) > MAX_PDF_PAGES:
        raise ParseError(f"PDF 페이지 수가 허용 한도({MAX_PDF_PAGES})를 초과했습니다.")
    pages: list[str] = []
    extracted_characters = 0
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = normalize_text(page.extract_text() or "")
        if page_text:
            extracted_characters += len(page_text)
            if extracted_characters > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                raise ParseError("PDF에서 추출한 텍스트가 허용 한도를 초과했습니다.")
            pages.append(f"[Page {page_number}]\n{page_text}")
    return ParsedDocument(
        text=normalize_text("\n\n".join(pages)),
        mime_type="application/pdf",
        metadata={"page_count": len(reader.pages)},
    )


def _parse_docx(payload: bytes) -> ParsedDocument:
    try:
        from docx import Document
    except ImportError as exc:  # pragma: no cover - dependency is declared
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
    """Extract text and tables from KS X 6101 HWPX (Hancom Word Open Container)."""
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
    """Extract text from supported files.

    Scanned PDFs and legacy binary HWP files need OCR/binary adapters and are
    deliberately reported as unsupported rather than silently indexed as garbage.
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

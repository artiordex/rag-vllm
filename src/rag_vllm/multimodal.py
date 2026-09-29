"""Multimodal document processing: Images (PNG/JPG), OCR, and Vision LLM interface."""

from __future__ import annotations

import base64
import logging
from io import BytesIO
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)


def parse_image_document(
    filename: str,
    payload: bytes,
    mime_type: str | None = None,
    ocr_lang: str = "kor+eng",
) -> tuple[str, dict[str, Any]]:
    """Parse image documents (PNG, JPEG, WEBP, etc.) with OCR & visual metadata.

    Returns:
        (extracted_text, metadata_dict)
    """
    try:
        img = Image.open(BytesIO(payload))
        width, height = img.size
        img_format = img.format or "IMAGE"
        mode = img.mode
    except Exception as exc:
        raise ValueError(f"손상된 이미지 파일입니다: {exc}") from exc

    metadata: dict[str, Any] = {
        "width": width,
        "height": height,
        "format": img_format,
        "mode": mode,
        "is_multimodal_image": True,
    }

    # 1. Try pytesseract OCR if installed and available
    ocr_text = ""
    try:
        import pytesseract

        ocr_text = pytesseract.image_to_string(img, lang=ocr_lang).strip()
        metadata["ocr_engine"] = "pytesseract"
    except (ImportError, Exception):
        metadata["ocr_engine"] = "none"

    if ocr_text:
        full_text = f"[이미지 문서 OCR 추출: {filename}]\n{ocr_text}"
        metadata["ocr_success"] = True
    else:
        # Structured descriptive placeholder for multimodal / visual indexing
        full_text = (
            f"[멀티모달 이미지 문서: {filename}]\n"
            f"- 규격: {width}x{height} 픽셀 ({img_format} 포맷, {mode} 모드)\n"
            f"- 안내: 이미지 내 표 및 시각 텍스트 정밀 인식을 위해 멀티모달 Vision LLM 연결을 지원합니다."
        )
        metadata["ocr_success"] = False

    return full_text, metadata


def encode_image_base64(payload: bytes, mime_type: str = "image/png") -> str:
    """Encode raw image bytes to data URI for Vision LLMs (e.g. Qwen2-VL / GPT-4o)."""
    b64 = base64.b64encode(payload).decode("ascii")
    return f"data:{mime_type};base64,{b64}"


def extract_visual_elements_from_pdf(payload: bytes) -> list[dict[str, Any]]:
    """Extract embedded images and diagrams from PDF bytes."""
    extracted = []
    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(payload))
        for page_idx, page in enumerate(reader.pages, start=1):
            if hasattr(page, "images"):
                for img_idx, img_file in enumerate(page.images, start=1):
                    extracted.append({
                        "page": page_idx,
                        "image_index": img_idx,
                        "name": getattr(img_file, "name", f"page_{page_idx}_img_{img_idx}"),
                        "data_size": len(img_file.data),
                    })
    except Exception as exc:
        logger.debug("PDF 내 이미지 추출 건너뜀: %s", exc)

    return extracted

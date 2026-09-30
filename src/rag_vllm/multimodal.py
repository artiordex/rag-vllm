# =============================================================================
# 파일명: multimodal.py
# 경로: src/rag_vllm/multimodal.py
# 목적: 이미지·PDF 시각 요소를 OCR과 Vision LLM 입력 형태로 변환함
# 작성자: AI전략팀
# 작성일: 2026-09-30
# 수정일: 2026-09-30
# =============================================================================

"""이미지·PDF 시각 요소를 OCR과 Vision LLM 입력 형태로 변환함"""

from __future__ import annotations

import base64
import logging
import warnings
from io import BytesIO
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)
MAX_IMAGE_PIXELS = 40_000_000


def parse_image_document(
    filename: str,
    payload: bytes,
    mime_type: str | None = None,
    ocr_lang: str = "kor+eng",
) -> tuple[str, dict[str, Any]]:
    """이미지 문서를 OCR하고 Vision LLM용 시각 메타데이터와 함께 반환함

    Args:
        filename: 원천 이미지 이름임
        payload: 이미지 바이트임
        mime_type: 업로드 MIME 타입임
        ocr_lang: Tesseract에 전달할 언어 조합임

    Returns:
        tuple[str, dict[str, Any]]: 추출 텍스트와 이미지 메타데이터임

    Raises:
        ValueError: 손상된 이미지이거나 픽셀 수 한도를 초과할 때 발생함
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            img = Image.open(BytesIO(payload))
        width, height = img.size
        if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
            raise ValueError(f"이미지 픽셀 수가 허용 한도({MAX_IMAGE_PIXELS})를 초과했습니다.")
        img.load()
        img_format = img.format or "IMAGE"
        mode = img.mode
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    except Exception as exc:
        raise ValueError("손상되었거나 허용 범위를 넘은 이미지 파일입니다.") from exc

    metadata: dict[str, Any] = {
        "width": width,
        "height": height,
        "format": img_format,
        "mode": mode,
        "is_multimodal_image": True,
    }

    # NOTE: Tesseract가 설치된 환경에서만 OCR을 시도하고 미설치도 유효한 경로로 취급함
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
        # NOTE: OCR이 없어도 이미지 크기·포맷을 색인해 후속 Vision LLM 연결 지점을 보존함
        full_text = (
            f"[멀티모달 이미지 문서: {filename}]\n"
            f"- 규격: {width}x{height} 픽셀 ({img_format} 포맷, {mode} 모드)\n"
            f"- 안내: 이미지 내 표 및 시각 텍스트 정밀 인식을 위해 멀티모달 Vision LLM 연결을 지원합니다."
        )
        metadata["ocr_success"] = False

    return full_text, metadata


def encode_image_base64(payload: bytes, mime_type: str = "image/png") -> str:
    """이미지 바이트를 Vision LLM이 받을 수 있는 data URI로 인코딩함

    Returns:
        str: MIME 타입과 base64 본문을 결합한 data URI임
    """
    b64 = base64.b64encode(payload).decode("ascii")
    return f"data:{mime_type};base64,{b64}"


def extract_visual_elements_from_pdf(payload: bytes) -> list[dict[str, Any]]:
    """PDF에 포함된 이미지의 페이지·순번·크기 메타데이터를 추출함

    Returns:
        list[dict[str, Any]]: 추출된 시각 요소 요약 목록임

    Caveats:
        PDF 이미지 추출 실패는 본문 파싱을 중단하지 않고 빈 목록으로 처리함
    """
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

# =============================================================================
# 파일명: moe_dq_client.py
# 경로: src/rag_vllm/moe_dq_client.py
# 목적: MOE 품질진단 개발 API와 비영속 방식으로 연동함
# 작성자: AI전략팀
# 작성일: 2026-09-30
# 수정일: 2026-09-30
# =============================================================================

"""MOE 품질진단 개발 API와 비영속 방식으로 연동함"""

from __future__ import annotations

import json
from typing import Any

import httpx

MAX_MOE_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_MOE_RESPONSE_BYTES = 5 * 1024 * 1024


class MoeDqUpstreamError(RuntimeError):
    """상위 MOE API 오류를 안전한 상태 코드와 메시지로 보존하는 예외임"""

    def __init__(self, status_code: int, detail: str) -> None:
        """상위 API 상태 코드와 안전한 오류 상세를 보관함"""
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


_OMITTED_KEYS = {
    "sampletext",
    "sourcetext",
    "rawtext",
    "textexcerpt",
    "originalfilename",
    "uploadedfilename",
    "rawbytes",
    "datafiedtext",
    "extractedtext",
    "content",
}


def _omit_source_fields(value: Any) -> Any:
    """상위 응답에서 원문·업로드 정보 필드를 재귀적으로 제거함

    Caveats:
        진단 결과를 사용자에게 전달할 때 원문이 실수로 재노출되지 않도록
        허용 목록이 아닌 제거 목록을 적용함
    """
    if isinstance(value, dict):
        return {
            key: _omit_source_fields(item)
            for key, item in value.items()
            if str(key).lower().replace("_", "").replace("-", "") not in _OMITTED_KEYS
        }
    if isinstance(value, list):
        return [_omit_source_fields(item) for item in value]
    return value


def _validate_unstructured_report(report: dict[str, Any]) -> None:
    """비정형 MOE 응답이 비영속·미승인 계약을 만족하는지 검증함

    Raises:
        MoeDqUpstreamError: 필수 메타데이터나 release 제한이 다를 때 발생함
    """
    metadata = report.get("metadata")
    assessment = metadata.get("unstructured_quality") if isinstance(metadata, dict) else None
    datafication = metadata.get("unstructured_datafication") if isinstance(metadata, dict) else None
    if (
        not isinstance(assessment, dict)
        or assessment.get("version") != "unstructured-quality-v1"
        or assessment.get("release_eligible") is not False
        or not isinstance(datafication, dict)
        or datafication.get("confidence_scope") != "block_classification_only"
        or datafication.get("transformation_accuracy") != "not_measured"
        or datafication.get("pii_clearance") != "not_assessed"
        or datafication.get("release_eligible") is not False
    ):
        raise MoeDqUpstreamError(502, "MOE 응답이 예상한 비정형 진단 계약과 일치하지 않습니다.")


def _validate_c2_report(report: dict[str, Any]) -> None:
    """C-2 용어 후보 응답이 읽기 전용·검토 대기 계약인지 검증함

    Raises:
        MoeDqUpstreamError: 원천 행 조회·DB 쓰기 금지 계약이 깨질 때 발생함
    """
    versions = report.get("dictionaryVersions")
    if (
        not isinstance(report.get("columns"), list)
        or not isinstance(versions, dict)
        or not all(isinstance(versions.get(key), str) and versions[key] for key in ("terms", "words"))
        or report.get("decisionStatus") != "suggestions-pending-domain-review"
        or report.get("sourceRowsRead") != 0
        or report.get("databaseWrites") != 0
    ):
        raise MoeDqUpstreamError(502, "MOE 응답이 예상한 읽기 전용 C-2 후보 계약과 일치하지 않습니다.")


async def _request_json(
    *,
    base_url: str,
    path: str,
    timeout_seconds: float,
    not_found_detail: str,
    method: str = "GET",
    params: dict[str, str] | None = None,
    files: dict[str, tuple[str, bytes, str]] | None = None,
) -> dict[str, Any]:
    """MOE API 응답을 크기·상태·JSON·원문 노출 기준으로 검증함

    Raises:
        MoeDqUpstreamError: 상위 API 상태, 응답 크기, JSON 구조가 허용 범위를
            벗어날 때 발생함
    """
    try:
        async with httpx.AsyncClient(
            timeout=timeout_seconds,
            follow_redirects=False,
            trust_env=False,
        ) as client:
            async with client.stream(
                method,
                f"{base_url}{path}",
                params=params,
                files=files,
            ) as response:
                if response.status_code == 413:
                    raise MoeDqUpstreamError(413, "MOE 개발 API 업로드 크기 한도를 초과했습니다.")
                if response.status_code in {400, 422}:
                    raise MoeDqUpstreamError(422, "MOE 개발 API가 입력 형식 또는 문서 파서 결과를 거부했습니다.")
                if response.status_code == 404:
                    raise MoeDqUpstreamError(404, not_found_detail)
                if not 200 <= response.status_code < 300:
                    raise MoeDqUpstreamError(502, "MOE 개발 API가 진단 요청을 처리하지 못했습니다.")

                declared_length = response.headers.get("content-length")
                if declared_length:
                    try:
                        if int(declared_length) > MAX_MOE_RESPONSE_BYTES:
                            raise MoeDqUpstreamError(502, "MOE 개발 API 응답 크기가 허용 한도를 초과했습니다.")
                    except ValueError:
                        pass

                response_body = bytearray()
                async for chunk in response.aiter_bytes(chunk_size=64 * 1024):
                    response_body.extend(chunk)
                    if len(response_body) > MAX_MOE_RESPONSE_BYTES:
                        raise MoeDqUpstreamError(502, "MOE 개발 API 응답 크기가 허용 한도를 초과했습니다.")
    except httpx.TimeoutException as exc:
        raise MoeDqUpstreamError(504, "MOE 개발 API 응답 시간이 초과되었습니다.") from exc
    except MoeDqUpstreamError:
        raise
    except httpx.ConnectError as exc:
        raise MoeDqUpstreamError(503, "MOE 개발 API에 연결할 수 없습니다.") from exc
    except httpx.HTTPError as exc:
        raise MoeDqUpstreamError(502, "MOE 개발 API에 연결하지 못했습니다.") from exc

    try:
        payload = json.loads(response_body)
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as exc:
        raise MoeDqUpstreamError(502, "MOE 개발 API가 JSON 응답을 반환하지 않았습니다.") from exc
    if not isinstance(payload, dict):
        raise MoeDqUpstreamError(502, "MOE 개발 API 응답 형식이 예상과 다릅니다.")
    try:
        sanitized = _omit_source_fields(payload)
    except RecursionError as exc:
        raise MoeDqUpstreamError(502, "MOE 개발 API 응답 중첩 깊이가 허용 범위를 넘었습니다.") from exc
    return sanitized


async def diagnose_unstructured_upload(
    *,
    base_url: str,
    timeout_seconds: float,
    filename: str,
    content_type: str,
    content: bytes,
) -> dict[str, Any]:
    """제한된 업로드를 MOE의 비영속 개발 진단 엔드포인트로 전달함

    Args:
        base_url: 인증정보가 포함되지 않은 MOE 기본 URL임
        timeout_seconds: 상위 요청 제한 시간임
        filename: 안전성 검증을 마친 업로드 파일명임
        content_type: 업로드 MIME 타입임
        content: 전달할 파일 바이트임

    Returns:
        dict[str, Any]: 저장하지 않은 개발 진단 결과임

    Raises:
        MoeDqUpstreamError: 업로드 크기·응답 계약·상위 API 호출이 실패할 때 발생함
    """
    if len(content) > MAX_MOE_UPLOAD_BYTES:
        raise MoeDqUpstreamError(413, "MOE 연동 업로드 한도(25 MiB)를 초과했습니다.")
    report = await _request_json(
        base_url=base_url,
        path="/api/v1/quality/documents/diagnose",
        timeout_seconds=timeout_seconds,
        not_found_detail="MOE 개발 진단 경로를 사용할 수 없습니다.",
        method="POST",
        files={"file": (filename, content, content_type)},
    )
    _validate_unstructured_report(report)
    return {
        "source": "moe_dq_diag",
        "assessment_scope": "development_unstructured_transient_upload",
        "stored_by_rag_vllm": False,
        "release_eligible": False,
        "report": report,
    }


async def get_c2_term_matches(
    *,
    base_url: str,
    timeout_seconds: float,
    table: str,
) -> dict[str, Any]:
    """기존 C-2 테이블의 용어 후보를 읽기 전용으로 조회함

    Returns:
        dict[str, Any]: 도메인 검토가 필요한 후보 결과임

    Raises:
        MoeDqUpstreamError: 상위 API 호출이나 읽기 전용 계약 검증에 실패할 때 발생함
    """
    report = await _request_json(
        base_url=base_url,
        path="/api/v1/standardization/term-matches",
        timeout_seconds=timeout_seconds,
        not_found_detail="MOE 개발 API 경로 또는 요청한 C-2 테이블을 찾지 못했습니다.",
        params={"table": table},
    )
    _validate_c2_report(report)
    return {
        "source": "moe_dq_diag",
        "assessment_scope": "existing_c2_column_names_only",
        "requires_domain_review": True,
        "release_eligible": False,
        "report": report,
    }

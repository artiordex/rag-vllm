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

import hashlib
import json
import math
import re
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
    "suggestion",
    "linenumber",
    "uploadedpath",
    "filepath",
}

_COMPARISON_DIMENSIONS = {
    "diagnostic_completeness",
    "diagnostic_integrity",
    "diagnostic_consistency",
    "diagnostic_privacy",
    "extractability",
    "structure_candidates",
    "structure_candidate_text_match",
    "transformation_accuracy",
    "structure_preservation",
    "deidentification_rights",
}
_COMPARISON_ISSUE_MESSAGE = "품질 규칙에 해당하는 이슈가 감지되었습니다."
_COMPARABLE_REASON = "동일 확장자·파서·OCR·규칙 버전으로 처리해 비교할 수 있습니다."
_INCOMPARABLE_REASONS = {
    "원본과 정제본의 파일 확장자가 달라 동일 기준으로 비교할 수 없습니다.",
    "원본과 정제본에 적용된 파서가 달라 동일 기준으로 비교할 수 없습니다.",
    "원본과 정제본의 OCR 엔진 버전이 달라 동일 기준으로 비교할 수 없습니다.",
    "원본과 정제본의 parser 버전이 달라 동일 기준으로 비교할 수 없습니다.",
    "원본과 정제본의 rules 버전이 달라 동일 기준으로 비교할 수 없습니다.",
    "원본과 정제본의 datafication 버전이 달라 동일 기준으로 비교할 수 없습니다.",
}
_QUALITY_CATEGORIES = {
    "COMPLETENESS": "완전성 (Completeness)",
    "INTEGRITY": "정확성/무결성 (Integrity)",
    "CONSISTENCY": "일관성/유효성 (Consistency)",
    "PRIVACY": "준수성/개인정보 (Privacy/Compliance)",
}
_QUALITY_RULE_CATEGORIES = {
    "CMP-001": "COMPLETENESS",
    "CMP-002": "COMPLETENESS",
    "CMP-003": "COMPLETENESS",
    "CMP-004": "COMPLETENESS",
    "INT-001": "INTEGRITY",
    "INT-002": "INTEGRITY",
    "INT-003": "INTEGRITY",
    "CNS-001": "CONSISTENCY",
    "CNS-002": "CONSISTENCY",
    "CNS-003": "CONSISTENCY",
    "CNS-004": "CONSISTENCY",
    "PRV-001": "PRIVACY",
    "PRV-002": "PRIVACY",
    "PRV-003": "PRIVACY",
}
_QUALITY_GRADES = {"A (우수)", "B (양호)", "C (보통/주의)", "D (미흡/개선필요)", "FAIL (부적격)"}
_ISSUE_SEVERITIES = {"INFO", "WARNING", "ERROR", "CRITICAL"}
_PARSERS_BY_EXTENSION = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".html": "html",
    ".xlsx": "xlsx",
    ".pptx": "pptx",
    ".hwpx": "hwpx",
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
}
_TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".log"}
_BLOCK_KINDS = {
    "heading",
    "numbered_clause",
    "list_item",
    "table_row",
    "table_separator",
    "paragraph",
}
_TARGET_FIELDS = {
    "notice": {
        "notice_id", "category", "title", "posted_date", "body_text", "attachments", "author_dept", "view_count"
    },
    "letter": {
        "letter_id", "title", "body_text", "registered_date", "attachments", "reply_required", "author",
        "source_format", "ocr_confidence", "pii_findings_count"
    },
    "schedule": {
        "academic_year", "event_id", "title", "event_type", "start_date", "end_date", "description"
    },
    "meal": {
        "meal_date", "meal_type", "menu_items", "nutrition", "allergen_codes", "image_url", "ocr_confidence"
    },
}
_TARGET_FIELD_STATUSES = {
    "structure_only",
    "not_observed",
    "metadata_candidate_only",
    "source_relation_required",
    "not_supported",
    "human_review_required",
}
_TARGET_FIELD_BASES = {
    "explicit_title_element_or_markdown_h1",
    "generic_paragraph_blocks_not_semantic_body",
    "unclassified_table_or_list_blocks",
    "filename_extension_only_not_verified_mime",
    "automatic_field_extraction_not_implemented",
    "attachment_or_image_relationship_not_in_upload",
    "not_measured_or_extracted_by_current_pipeline",
}
_MAX_DIAGNOSTIC_CHARS = 5_000_000
_MAX_DOCUMENT_BYTES = 25 * 1024 * 1024


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
        or not _validate_diagnostic_report(report)
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


def _contains_source_field(value: Any) -> bool:
    """중첩된 MOE JSON에 원문·경로·위치 필드가 다시 나타나면 거부함"""
    if isinstance(value, dict):
        return any(
            str(key).lower().replace("_", "").replace("-", "") in _OMITTED_KEYS
            or _contains_source_field(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_source_field(item) for item in value)
    return False


def _is_count(value: Any, *, maximum: int = _MAX_DIAGNOSTIC_CHARS) -> bool:
    return type(value) is int and 0 <= value <= maximum


def _is_number(value: Any, *, minimum: float, maximum: float) -> bool:
    return (
        type(value) in {int, float}
        and math.isfinite(float(value))
        and minimum <= float(value) <= maximum
    )


def _is_bounded_text(value: Any, *, maximum: int, nonempty: bool = True) -> bool:
    return (
        isinstance(value, str)
        and len(value) <= maximum
        and (bool(value) or not nonempty)
        and value.isprintable()
    )


def _validate_quality_profile(value: Any) -> bool:
    if (
        not isinstance(value, dict)
        or set(value) != {"version", "dimensions", "release_eligible"}
        or value.get("version") != "unstructured-quality-v1"
        or value.get("release_eligible") is not False
    ):
        return False
    dimensions = value.get("dimensions")
    if not isinstance(dimensions, dict) or set(dimensions) != {
        "extractability", "transformation_accuracy", "structure_preservation", "deidentification_rights"
    }:
        return False

    extractability = dimensions["extractability"]
    extract_keys = {
        "status", "result", "extracted_characters", "page_count", "ocr_status", "ocr_engine",
        "ocr_word_count", "ocr_mean_word_confidence", "ocr_confidence_semantics", "note"
    }
    if (
        not isinstance(extractability, dict)
        or not {"status", "result", "extracted_characters", "note"}.issubset(extractability)
        or not set(extractability).issubset(extract_keys)
        or extractability.get("status") != "measured"
        or not isinstance(extractability.get("result"), str)
        or extractability.get("result") not in {"text_found", "no_text"}
        or not _is_count(extractability.get("extracted_characters"))
        or extractability.get("note") != "파일 1건의 추출 결과이며 모집단 파싱 성공률은 아닙니다."
    ):
        return False
    page_count = extractability.get("page_count")
    if page_count is not None and not _is_count(page_count, maximum=2_000):
        return False
    ocr_status = extractability.get("ocr_status")
    if ocr_status is not None and (
        not isinstance(ocr_status, str)
        or ocr_status not in {"completed", "no_text", "not_run", "not_applicable"}
    ):
        return False
    ocr_engine = extractability.get("ocr_engine")
    if ocr_engine is not None and ocr_engine != "tesseract":
        return False
    ocr_word_count = extractability.get("ocr_word_count")
    if ocr_word_count is not None and not _is_count(ocr_word_count, maximum=200_000):
        return False
    confidence = extractability.get("ocr_mean_word_confidence")
    if confidence is not None and not _is_number(confidence, minimum=0, maximum=100):
        return False
    semantics = extractability.get("ocr_confidence_semantics")
    if semantics is not None and semantics != "tesseract_word_confidence_not_accuracy":
        return False

    fixed_not_measured = {
        "transformation_accuracy": (
            "승인된 정답셋과 OCR 결과 대조가 없어 정확도는 산출하지 않았습니다.",
            {"status", "cer", "wer", "processing_accuracy", "note"},
        ),
        "structure_preservation": (
            "현재 파서 계약은 평문이며 제목·문단·표 구조 보존을 측정하지 않습니다.",
            {"status", "tag_preservation_rate", "table_shape_match_rate", "parser_output", "note"},
        ),
    }
    for name, (note, allowed_keys) in fixed_not_measured.items():
        dimension = dimensions[name]
        if (
            not isinstance(dimension, dict)
            or not {"status", "note"}.issubset(dimension)
            or not set(dimension).issubset(allowed_keys)
            or dimension.get("status") != "not_measured"
            or dimension.get("note") != note
            or any(
                dimension.get(key) is not None
                for key in allowed_keys - {"status", "note", "parser_output"}
            )
        ):
            return False
        if name == "structure_preservation" and dimension.get("parser_output", "flattened_text") != "flattened_text":
            return False

    rights = dimensions["deidentification_rights"]
    if (
        not isinstance(rights, dict)
        or set(rights) != {"status", "pii_scan", "masking_status", "copyright_status", "note"}
        or rights.get("status") != "review_required"
        or rights.get("pii_scan") != "heuristic_only"
        or rights.get("masking_status") != "not_performed"
        or rights.get("copyright_status") != "unknown"
        or rights.get("note") != "규칙 기반 이슈 검사는 마스킹이나 저작권 판정을 대신하지 않습니다."
    ):
        return False
    return True


def _validate_datafication_summary(value: Any) -> bool:
    expected_keys = {
        "algorithm_version", "candidate_block_count", "candidate_block_counts", "candidate_cell_count",
        "table_group_count", "page_mapped_block_count", "unmapped_block_count", "title_candidate_present",
        "page_mapping", "confidence_scope", "transformation_accuracy", "pii_clearance", "release_eligible"
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected_keys
        or value.get("algorithm_version") != "unstructured-datafication-v1"
        or not _is_count(value.get("candidate_block_count"))
        or not _is_count(value.get("candidate_cell_count"))
        or not _is_count(value.get("table_group_count"))
        or not _is_count(value.get("page_mapped_block_count"))
        or not _is_count(value.get("unmapped_block_count"))
        or not isinstance(value.get("title_candidate_present"), bool)
        or not isinstance(value.get("page_mapping"), str)
        or value.get("page_mapping") not in {"parser_page_character_ranges", "unavailable_from_flattened_parser_output"}
        or value.get("confidence_scope") != "block_classification_only"
        or value.get("transformation_accuracy") != "not_measured"
        or value.get("pii_clearance") != "not_assessed"
        or value.get("release_eligible") is not False
    ):
        return False
    counts = value.get("candidate_block_counts")
    return (
        isinstance(counts, dict)
        and set(counts) == _BLOCK_KINDS
        and all(_is_count(count) for count in counts.values())
        and sum(counts.values()) == value["candidate_block_count"]
        and value["page_mapped_block_count"] + value["unmapped_block_count"] == value["candidate_block_count"]
    )


def _validate_target_summary(value: Any) -> bool:
    if (
        not isinstance(value, dict)
        or set(value) != {"algorithm_version", "dataset_kind_selection_required", "field_values_returned", "persisted", "datasets"}
        or value.get("algorithm_version") != "unstructured-datafication-v1"
        or value.get("dataset_kind_selection_required") is not True
        or value.get("field_values_returned") is not False
        or value.get("persisted") is not False
        or not isinstance(value.get("datasets"), list)
        or len(value["datasets"]) != len(_TARGET_FIELDS)
    ):
        return False
    labels = {"notice": "공지사항", "letter": "가정통신문", "schedule": "학사일정", "meal": "식단"}
    seen: set[str] = set()
    for dataset in value["datasets"]:
        if (
            not isinstance(dataset, dict)
            or set(dataset) != {"dataset_kind", "label", "fields"}
            or not isinstance(dataset.get("dataset_kind"), str)
            or dataset.get("dataset_kind") not in _TARGET_FIELDS
        ):
            return False
        kind = dataset["dataset_kind"]
        fields = dataset.get("fields")
        if (
            kind in seen
            or dataset.get("label") != labels[kind]
            or not isinstance(fields, list)
            or len(fields) != len(_TARGET_FIELDS[kind])
        ):
            return False
        seen.add(kind)
        field_names: set[str] = set()
        for field in fields:
            if (
                not isinstance(field, dict)
                or set(field) != {"field_name", "status", "candidate_count", "basis"}
                or not isinstance(field.get("field_name"), str)
                or field.get("field_name") not in _TARGET_FIELDS[kind]
                or not isinstance(field.get("status"), str)
                or field.get("status") not in _TARGET_FIELD_STATUSES
                or not _is_count(field.get("candidate_count"))
                or not isinstance(field.get("basis"), str)
                or field.get("basis") not in _TARGET_FIELD_BASES
            ):
                return False
            field_names.add(field["field_name"])
        if field_names != _TARGET_FIELDS[kind]:
            return False
    return seen == set(_TARGET_FIELDS)


def _validate_safe_metadata(
    metadata: Any,
    *,
    expected_extension: str | None = None,
    expected_size: int | None = None,
) -> bool:
    if not isinstance(metadata, dict):
        return False
    allowed_keys = {
        "extension", "file_size_bytes", "parser", "extracted_characters", "extraction_status", "page_count",
        "ocr_engine", "ocr_engine_version", "ocr_language", "ocr_status", "ocr_word_count",
        "ocr_mean_word_confidence", "unstructured_quality", "unstructured_datafication",
        "unstructured_datafication_targets"
    }
    required = {
        "extension", "file_size_bytes", "parser", "extracted_characters", "extraction_status",
        "unstructured_quality", "unstructured_datafication", "unstructured_datafication_targets"
    }
    extension = metadata.get("extension")
    if (
        not required.issubset(metadata)
        or not set(metadata).issubset(allowed_keys)
        or not isinstance(extension, str)
        or extension not in _TEXT_EXTENSIONS | set(_PARSERS_BY_EXTENSION)
        or (expected_extension is not None and extension != expected_extension)
    ):
        return False
    expected_parser = _PARSERS_BY_EXTENSION.get(extension, "text")
    file_size = metadata.get("file_size_bytes")
    extracted_chars = metadata.get("extracted_characters")
    quality_profile = metadata.get("unstructured_quality")
    if (
        not _is_count(file_size, maximum=_MAX_DOCUMENT_BYTES)
        or file_size == 0
        or (expected_size is not None and file_size != expected_size)
        or metadata.get("parser") != expected_parser
        or not _is_count(extracted_chars)
        or not isinstance(metadata.get("extraction_status"), str)
        or metadata.get("extraction_status") not in {"text_found", "no_text"}
        or not _validate_quality_profile(quality_profile)
        or quality_profile["dimensions"]["extractability"].get("result")
        != metadata.get("extraction_status")
        or not _validate_datafication_summary(metadata.get("unstructured_datafication"))
        or not _validate_target_summary(metadata.get("unstructured_datafication_targets"))
    ):
        return False

    if "page_count" in metadata:
        if extension != ".pdf" or not _is_count(metadata["page_count"], maximum=2_000):
            return False
    elif extension == ".pdf":
        return False
    ocr_keys = {"ocr_engine", "ocr_engine_version", "ocr_language", "ocr_status", "ocr_word_count", "ocr_mean_word_confidence"}
    if set(metadata).intersection(ocr_keys):
        if expected_parser not in {"image", "pdf"}:
            return False
        if metadata.get("ocr_engine") != "tesseract" or metadata.get("ocr_language") != "kor+eng":
            return False
        if not _is_bounded_text(metadata.get("ocr_engine_version"), maximum=64):
            return False
        if not isinstance(metadata.get("ocr_status"), str) or metadata.get("ocr_status") not in {"completed", "no_text"}:
            return False
        if not _is_count(metadata.get("ocr_word_count"), maximum=200_000):
            return False
        confidence = metadata.get("ocr_mean_word_confidence")
        if confidence is not None and not _is_number(confidence, minimum=0, maximum=100):
            return False
        if expected_parser == "image" and ((metadata["ocr_status"] == "no_text") != (extracted_chars == 0)):
            return False
    return quality_profile["dimensions"]["extractability"]["extracted_characters"] == extracted_chars


def _validate_diagnostic_report(
    report: Any,
    *,
    expected_extension: str | None = None,
    expected_size: int | None = None,
) -> bool:
    expected_keys = {
        "document_name", "total_chars", "total_lines", "overall_score", "grade", "passed",
        "category_scores", "issues", "diagnosed_at", "metadata"
    }
    if (
        not isinstance(report, dict)
        or set(report) != expected_keys
        or not isinstance(report.get("metadata"), dict)
        or not isinstance(report.get("issues"), list)
        or not _validate_safe_metadata(
            report["metadata"], expected_extension=expected_extension, expected_size=expected_size
        )
        or report.get("document_name") != f"uploaded_document{report['metadata']['extension']}"
        or not _is_count(report.get("total_chars"))
        or report.get("total_chars") != report["metadata"].get("extracted_characters")
        or not _is_count(report.get("total_lines"), maximum=_MAX_DIAGNOSTIC_CHARS + 1)
        or report["total_lines"] > report["total_chars"] + 1
        or not _is_number(report.get("overall_score"), minimum=0, maximum=100)
        or not isinstance(report.get("grade"), str)
        or report.get("grade") not in _QUALITY_GRADES
        or not isinstance(report.get("passed"), bool)
        or not _is_bounded_text(report.get("diagnosed_at"), maximum=40)
    ):
        return False

    scores = report.get("category_scores")
    if not isinstance(scores, dict) or set(scores) != set(_QUALITY_CATEGORIES):
        return False
    issue_counts = {category: 0 for category in _QUALITY_CATEGORIES}
    for issue in report["issues"]:
        if (
            not isinstance(issue, dict)
            or set(issue) != {"rule_id", "category", "severity", "message"}
            or not isinstance(issue.get("rule_id"), str)
            or issue.get("rule_id") not in _QUALITY_RULE_CATEGORIES
            or issue.get("category") != _QUALITY_CATEGORIES[_QUALITY_RULE_CATEGORIES[issue["rule_id"]]]
            or not isinstance(issue.get("severity"), str)
            or issue.get("severity") not in _ISSUE_SEVERITIES
            or issue.get("message") != _COMPARISON_ISSUE_MESSAGE
        ):
            return False
        issue_counts[_QUALITY_RULE_CATEGORIES[issue["rule_id"]]] += 1
    for name, score in scores.items():
        if (
            not isinstance(score, dict)
            or set(score) != {"category", "score", "issue_count", "description"}
            or score.get("category") != _QUALITY_CATEGORIES[name]
            or not _is_number(score.get("score"), minimum=0, maximum=100)
            or not _is_count(score.get("issue_count"))
            or score["issue_count"] != issue_counts[name]
            or not _is_bounded_text(score.get("description"), maximum=128)
        ):
            return False
        issue_count = score["issue_count"]
        category_score = float(score["score"])
        expected_description = (
            "결함 없음 (우수)" if issue_count == 0 else
            f"{issue_count}건의 경미한 이상 발견 (양호)" if category_score >= 80 else
            f"{issue_count}건의 품질 결함 검출 (주의/개선필요)" if category_score >= 60 else
            f"{issue_count}건의 심각한 결함 검출 (부적격/조치필요)"
        )
        if score["description"] != expected_description:
            return False
    return not _contains_source_field(report)


def _validate_comparison_side(
    side: Any,
    *,
    expected_digest: str,
    expected_size: int,
    expected_extension: str,
) -> bool:
    """비교 응답의 파서 측정값·진단 요약과 원문 비노출 조건을 검증함"""
    if (
        not isinstance(side, dict)
        or set(side) != {"file", "report"}
        or not isinstance(side.get("file"), dict)
        or not isinstance(side.get("report"), dict)
    ):
        return False
    file_info = side["file"]
    if (
        not {"sha256", "file_size_bytes", "extension", "parser"}.issubset(file_info)
        or not set(file_info).issubset({"sha256", "file_size_bytes", "extension", "parser", "ocr_engine_version"})
        or not isinstance(file_info.get("sha256"), str)
        or not re.fullmatch(r"[a-f0-9]{64}", file_info["sha256"])
        or file_info.get("sha256") != expected_digest
        or not _is_count(file_info.get("file_size_bytes"), maximum=_MAX_DOCUMENT_BYTES)
        or file_info.get("file_size_bytes") != expected_size
        or file_info.get("extension") != expected_extension
        or expected_extension not in _TEXT_EXTENSIONS | set(_PARSERS_BY_EXTENSION)
        or file_info.get("parser") != _PARSERS_BY_EXTENSION.get(expected_extension, "text")
        or (
            file_info.get("ocr_engine_version") is not None
            and not _is_bounded_text(file_info.get("ocr_engine_version"), maximum=64)
        )
        or file_info.get("ocr_engine_version")
        != side["report"].get("metadata", {}).get("ocr_engine_version")
    ):
        return False
    if (
        not _validate_diagnostic_report(
            side["report"], expected_extension=expected_extension, expected_size=expected_size
        )
    ):
        return False
    return True


def _validate_side_dimensions(side: Any, *, comparable: bool) -> bool:
    if not isinstance(side, dict):
        return False
    diagnostic_keys = {f"diagnostic_{category.lower()}" for category in _QUALITY_CATEGORIES}
    for key in diagnostic_keys:
        dimension = side.get(key)
        if (
            not isinstance(dimension, dict)
            or not {"status", "original", "cleaned"}.issubset(dimension)
            or not set(dimension).issubset({"status", "original", "cleaned", "score_delta", "issue_count_delta"})
            or dimension.get("status") != "measured"
            or not isinstance(dimension.get("original"), dict)
            or not isinstance(dimension.get("cleaned"), dict)
            or set(dimension["original"]) != {"score", "issue_count"}
            or set(dimension["cleaned"]) != {"score", "issue_count"}
            or not _is_number(dimension["original"].get("score"), minimum=0, maximum=100)
            or not _is_number(dimension["cleaned"].get("score"), minimum=0, maximum=100)
            or not _is_count(dimension["original"].get("issue_count"))
            or not _is_count(dimension["cleaned"].get("issue_count"))
        ):
            return False
        score_delta = dimension.get("score_delta")
        count_delta = dimension.get("issue_count_delta")
        if comparable:
            if (
                not _is_number(score_delta, minimum=-100, maximum=100)
                or score_delta != round(float(dimension["cleaned"]["score"]) - float(dimension["original"]["score"]), 1)
                or type(count_delta) is not int
                or count_delta != dimension["cleaned"]["issue_count"] - dimension["original"]["issue_count"]
            ):
                return False
        elif score_delta is not None or count_delta is not None:
            return False

    extractability = side.get("extractability")
    if (
        not isinstance(extractability, dict)
        or not {"status", "original", "cleaned", "ocr_confidence_semantics"}.issubset(extractability)
        or not set(extractability).issubset({
            "status", "original", "cleaned", "extracted_character_delta", "ocr_confidence_semantics"
        })
        or extractability.get("status") != "measured"
        or extractability.get("ocr_confidence_semantics") != "Tesseract confidence is not transcription accuracy."
    ):
        return False
    for key in ("original", "cleaned"):
        value = extractability.get(key)
        if (
            not isinstance(value, dict)
            or not {"result", "extracted_characters"}.issubset(value)
            or not set(value).issubset({
                "result", "extracted_characters", "page_count", "ocr_status", "ocr_word_count",
                "ocr_mean_word_confidence"
            })
            or not isinstance(value.get("result"), str)
            or value.get("result") not in {"text_found", "no_text"}
            or not _is_count(value.get("extracted_characters"))
            or (value.get("page_count") is not None and not _is_count(value.get("page_count"), maximum=2_000))
            or (
                value.get("ocr_status") is not None
                and (
                    not isinstance(value.get("ocr_status"), str)
                    or value.get("ocr_status") not in {"completed", "no_text", "not_run", "not_applicable"}
                )
            )
            or (value.get("ocr_word_count") is not None and not _is_count(value.get("ocr_word_count"), maximum=200_000))
            or (
                value.get("ocr_mean_word_confidence") is not None
                and not _is_number(value.get("ocr_mean_word_confidence"), minimum=0, maximum=100)
            )
        ):
            return False
    char_delta = extractability.get("extracted_character_delta")
    if comparable:
        expected_delta = extractability["cleaned"]["extracted_characters"] - extractability["original"]["extracted_characters"]
        if type(char_delta) is not int or char_delta != expected_delta:
            return False
    elif char_delta is not None:
        return False

    structure = side.get("structure_candidates")
    count_keys = {"candidate_block_count", "candidate_cell_count", "table_group_count"}
    if (
        not isinstance(structure, dict)
        or not {"status", "original", "cleaned", "note"}.issubset(structure)
        or not set(structure).issubset({"status", "original", "cleaned", "delta", "note"})
        or structure.get("status") != "measured"
        or structure.get("note") != "파서 출력에서 찾은 구조 후보의 개수이며 원본 레이아웃 보존을 증명하지 않습니다."
    ):
        return False
    for key in ("original", "cleaned"):
        value = structure.get(key)
        if (
            not isinstance(value, dict)
            or set(value) != count_keys | {"candidate_block_counts"}
            or any(not _is_count(value.get(count_key)) for count_key in count_keys)
            or not isinstance(value.get("candidate_block_counts"), dict)
            or set(value["candidate_block_counts"]) != _BLOCK_KINDS
            or not all(_is_count(count) for count in value["candidate_block_counts"].values())
            or sum(value["candidate_block_counts"].values()) != value["candidate_block_count"]
        ):
            return False
    delta = structure.get("delta")
    if comparable:
        if (
            not isinstance(delta, dict)
            or set(delta) != count_keys | {"candidate_block_counts"}
            or any(
                type(delta.get(key)) is not int
                or delta[key] != structure["cleaned"][key] - structure["original"][key]
                for key in count_keys
            )
            or not isinstance(delta.get("candidate_block_counts"), dict)
            or set(delta["candidate_block_counts"]) != _BLOCK_KINDS
            or any(
                type(delta["candidate_block_counts"].get(key)) is not int
                or delta["candidate_block_counts"][key]
                != structure["cleaned"]["candidate_block_counts"][key] - structure["original"]["candidate_block_counts"][key]
                for key in _BLOCK_KINDS
            )
        ):
            return False
    elif delta is not None:
        return False

    text_match = side.get("structure_candidate_text_match")
    if (
        not isinstance(text_match, dict)
        or not {"status", "note"}.issubset(text_match)
        or not set(text_match).issubset({"status", "values", "note"})
        or text_match.get("status") != ("measured" if comparable else "unmeasured")
        or text_match.get("note") != "동일 종류의 후보 블록에서 정규화 텍스트가 정확히 같은 개수입니다. 의미 유사도나 전체 구조 정확도는 아닙니다."
    ):
        return False
    match_values = text_match.get("values")
    match_keys = {
        "original_candidate_count", "cleaned_candidate_count", "exact_matching_candidate_count",
        "removed_candidate_count", "introduced_candidate_count"
    }
    if comparable:
        if (
            not isinstance(match_values, dict)
            or set(match_values) != match_keys
            or any(not _is_count(match_values.get(key)) for key in match_keys)
            or match_values["exact_matching_candidate_count"] > min(
                match_values["original_candidate_count"], match_values["cleaned_candidate_count"]
            )
            or match_values["removed_candidate_count"] != (
                match_values["original_candidate_count"] - match_values["exact_matching_candidate_count"]
            )
            or match_values["introduced_candidate_count"] != (
                match_values["cleaned_candidate_count"] - match_values["exact_matching_candidate_count"]
            )
        ):
            return False
    elif match_values is not None:
        return False

    for key, note in (
        ("transformation_accuracy", "승인된 정답셋 대조가 없어 CER/WER 또는 가공 정확도를 산출하지 않습니다."),
        ("structure_preservation", "평탄화 파서 출력만으로 원본의 전체 제목·문단·표 레이아웃 보존율을 산출하지 않습니다."),
    ):
        value = side.get(key)
        if not isinstance(value, dict) or set(value) != {"status", "reason"} or value.get("status") != "not_measured" or value.get("reason") != note:
            return False

    rights = side.get("deidentification_rights")
    if (
        not isinstance(rights, dict)
        or set(rights) != {"status", "original", "cleaned", "reason"}
        or rights.get("status") != "review_required"
        or rights.get("reason") != "규칙 기반 패턴 진단은 개인정보 마스킹이나 저작권 검수를 대신하지 않습니다."
    ):
        return False
    expected_rights_side = {"pii_scan": "heuristic_only", "masking_status": "not_performed", "copyright_status": "unknown"}
    if rights.get("original") != expected_rights_side or rights.get("cleaned") != expected_rights_side:
        return False
    return True


def _validate_document_comparison(
    report: dict[str, Any],
    *,
    original_digest: str,
    original_size: int,
    original_extension: str,
    cleaned_digest: str,
    cleaned_size: int,
    cleaned_extension: str,
) -> None:
    """비정형 원본·정제본 비교 계약과 측정·비측정 상태를 확인함"""
    expected_keys = {
        "contract_version", "comparable", "comparability_reason", "versions", "original", "cleaned",
        "dimensions", "issue_delta"
    }
    if (
        set(report) != expected_keys
        or report.get("contract_version") != "document-comparison-v1"
        or type(report.get("comparable")) is not bool
        or not _is_bounded_text(report.get("comparability_reason"), maximum=256)
        or not isinstance(report.get("versions"), dict)
        or set(report["versions"]) != {"parser", "rules", "datafication"}
        or not all(_is_bounded_text(value, maximum=128) for value in report["versions"].values())
        or not isinstance(report.get("dimensions"), dict)
        or set(report["dimensions"]) != _COMPARISON_DIMENSIONS
        or not isinstance(report.get("issue_delta"), dict)
        or not _validate_comparison_side(
            report.get("original"),
            expected_digest=original_digest,
            expected_size=original_size,
            expected_extension=original_extension,
        )
        or not _validate_comparison_side(
            report.get("cleaned"),
            expected_digest=cleaned_digest,
            expected_size=cleaned_size,
            expected_extension=cleaned_extension,
        )
        or report["comparable"] != (
            original_extension == cleaned_extension
            and report["original"]["file"]["parser"] == report["cleaned"]["file"]["parser"]
            and report["original"]["file"].get("ocr_engine_version")
            == report["cleaned"]["file"].get("ocr_engine_version")
        )
        or (
            report["comparable"]
            and report["comparability_reason"] != _COMPARABLE_REASON
        )
        or (
            not report["comparable"]
            and report["comparability_reason"] not in _INCOMPARABLE_REASONS
        )
        or not _validate_side_dimensions(report["dimensions"], comparable=report["comparable"])
    ):
        raise MoeDqUpstreamError(502, "MOE 응답이 예상한 원본·정제본 비교 계약과 일치하지 않습니다.")

    dimensions = report["dimensions"]
    for category in _QUALITY_CATEGORIES:
        dimension = dimensions[f"diagnostic_{category.lower()}"]
        original_score = report["original"]["report"]["category_scores"][category]
        cleaned_score = report["cleaned"]["report"]["category_scores"][category]
        if (
            dimension["original"]["score"] != original_score["score"]
            or dimension["original"]["issue_count"] != original_score["issue_count"]
            or dimension["cleaned"]["score"] != cleaned_score["score"]
            or dimension["cleaned"]["issue_count"] != cleaned_score["issue_count"]
        ):
            raise MoeDqUpstreamError(502, "MOE 응답의 진단 요약이 개별 진단 보고서와 일치하지 않습니다.")
    for side_name in ("original", "cleaned"):
        dimension_side = "original" if side_name == "original" else "cleaned"
        metadata = report[side_name]["report"]["metadata"]
        if (
            dimensions["extractability"][dimension_side]["extracted_characters"]
            != metadata["extracted_characters"]
            or dimensions["extractability"][dimension_side]["result"]
            != metadata["extraction_status"]
            or report["versions"]["datafication"]
            != metadata["unstructured_datafication"]["algorithm_version"]
        ):
            raise MoeDqUpstreamError(502, "MOE 응답의 추출 요약이 개별 진단 보고서와 일치하지 않습니다.")

    structure = dimensions["structure_candidates"]
    for side_name in ("original", "cleaned"):
        summary = report[side_name]["report"]["metadata"]["unstructured_datafication"]
        side_summary = structure[side_name]
        for key in ("candidate_block_count", "candidate_cell_count", "table_group_count"):
            if side_summary[key] != summary[key]:
                raise MoeDqUpstreamError(502, "MOE 응답의 구조 요약이 개별 진단 보고서와 일치하지 않습니다.")
        if side_summary["candidate_block_counts"] != summary["candidate_block_counts"]:
            raise MoeDqUpstreamError(502, "MOE 응답의 구조 유형 요약이 개별 진단 보고서와 일치하지 않습니다.")
    if report["comparable"]:
        text_match_values = dimensions["structure_candidate_text_match"]["values"]
        if (
            text_match_values["original_candidate_count"]
            != structure["original"]["candidate_block_count"]
            or text_match_values["cleaned_candidate_count"]
            != structure["cleaned"]["candidate_block_count"]
        ):
            raise MoeDqUpstreamError(502, "MOE 응답의 후보 일치 요약이 구조 후보 집계와 일치하지 않습니다.")

    delta = report["issue_delta"]
    expected_delta_fields = {"status", "resolved", "introduced", "remaining", "unmeasured", "reason"}
    if (
        set(delta) != expected_delta_fields
        or delta.get("status") != ("measured" if report["comparable"] else "unmeasured")
        or not _is_bounded_text(delta.get("reason"), maximum=256)
        or any(
            not isinstance(delta.get(key), list)
            or not all(isinstance(rule_id, str) and rule_id in _QUALITY_RULE_CATEGORIES for rule_id in delta[key])
            or delta[key] != sorted(set(delta[key]))
            for key in ("resolved", "introduced", "remaining", "unmeasured")
        )
        or _contains_source_field(report)
    ):
        raise MoeDqUpstreamError(502, "MOE 응답의 이슈 변화 또는 원문 비노출 조건이 유효하지 않습니다.")

    original_rules = {issue["rule_id"] for issue in report["original"]["report"]["issues"]}
    cleaned_rules = {issue["rule_id"] for issue in report["cleaned"]["report"]["issues"]}
    if report["comparable"]:
        expected_lists = {
            "resolved": sorted(original_rules - cleaned_rules),
            "introduced": sorted(cleaned_rules - original_rules),
            "remaining": sorted(original_rules & cleaned_rules),
            "unmeasured": [],
        }
        valid_reason = delta["reason"] == "동일 확장자·파서·규칙 버전에서 규칙 ID 존재 여부를 비교했습니다."
    else:
        expected_lists = {
            "resolved": [],
            "introduced": [],
            "remaining": [],
            "unmeasured": sorted(original_rules | cleaned_rules),
        }
        valid_reason = delta["reason"] == report["comparability_reason"]
    if any(delta[key] != value for key, value in expected_lists.items()) or not valid_reason:
        raise MoeDqUpstreamError(502, "MOE 응답의 규칙 변화가 개별 진단 보고서와 일치하지 않습니다.")


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


async def compare_unstructured_uploads(
    *,
    base_url: str,
    timeout_seconds: float,
    original_extension: str,
    original_content_type: str,
    original_content: bytes,
    cleaned_extension: str,
    cleaned_content_type: str,
    cleaned_content: bytes,
) -> dict[str, Any]:
    """원본과 정제본을 MOE 개발 비교 API에 임시 전달하고 측정 계약을 확인함"""
    max_file_bytes = MAX_MOE_UPLOAD_BYTES
    if len(original_content) > max_file_bytes or len(cleaned_content) > max_file_bytes:
        raise MoeDqUpstreamError(413, "원본 또는 정제본이 MOE 비교 업로드 한도(25 MiB)를 초과했습니다.")
    for extension in (original_extension, cleaned_extension):
        if not re.fullmatch(r"\.[a-z0-9]{1,15}", extension):
            raise MoeDqUpstreamError(422, "원본 또는 정제본의 파일 확장자가 올바르지 않습니다.")
    if not original_content or not cleaned_content:
        raise MoeDqUpstreamError(422, "원본과 정제본은 빈 파일일 수 없습니다.")

    original_digest = hashlib.sha256(original_content).hexdigest()
    cleaned_digest = hashlib.sha256(cleaned_content).hexdigest()
    report = await _request_json(
        base_url=base_url,
        path="/api/v1/quality/documents/compare",
        timeout_seconds=timeout_seconds,
        not_found_detail="MOE 개발 문서 비교 경로를 사용할 수 없습니다.",
        method="POST",
        files={
            "original_file": (f"upload{original_extension}", original_content, original_content_type),
            "cleaned_file": (f"upload{cleaned_extension}", cleaned_content, cleaned_content_type),
        },
    )
    _validate_document_comparison(
        report,
        original_digest=original_digest,
        original_size=len(original_content),
        original_extension=original_extension,
        cleaned_digest=cleaned_digest,
        cleaned_size=len(cleaned_content),
        cleaned_extension=cleaned_extension,
    )
    return {
        "source": "moe_dq_diag",
        "assessment_scope": "development_transient_original_cleaned_comparison",
        "stored_by_rag_vllm": False,
        "release_eligible": False,
        "comparison": report,
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

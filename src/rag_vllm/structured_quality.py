# =============================================================================
# 파일명: structured_quality.py
# 경로: src/rag_vllm/structured_quality.py
# 목적: CSV·테이블·JSON 레코드의 설명 가능한 구조 품질 점검 수행함
# 작성자: AI전략팀
# 작성일: 2026-09-30
# 수정일: 2026-09-30
# =============================================================================

"""CSV·테이블·JSON 레코드의 설명 가능한 구조 품질 점검 수행함

공식 공공데이터 품질평가나 공개 여부 판정이 아닌 로컬 규칙으로 동작함
"""

from __future__ import annotations

import csv
from datetime import date
import io
import re
from typing import Any

from .chunking import MAX_TEXT_CHARACTERS

DATE_PATTERN = re.compile(r"^\d{4}([-/.])\d{2}\1\d{2}$")
PHONE_PATTERN = re.compile(r"^01[016789]-?\d{3,4}-?\d{4}$")
EMAIL_PATTERN = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
MAX_STRUCTURED_ROWS = 100_000
MAX_STRUCTURED_COLUMNS = 512
MAX_STRUCTURED_CELLS = 1_000_000


def _is_valid_date(value: str) -> bool:
    """지원하는 날짜 구분자 형식과 실제 달력 날짜 여부를 함께 확인함"""
    if not DATE_PATTERN.fullmatch(value):
        return False
    normalized = value.replace("/", "-").replace(".", "-")
    try:
        date.fromisoformat(normalized)
        return True
    except ValueError:
        return False


def evaluate_structured_dataset(
    records: list[dict[str, Any]] | None = None,
    csv_text: str | None = None,
    primary_key_col: str | None = None,
) -> dict[str, Any]:
    """CSV 또는 레코드 목록의 구조 품질 신호를 로컬 규칙으로 계산함

    Args:
        records: JSON 레코드 목록임
        csv_text: CSV 원문 입력임
        primary_key_col: 중복을 확인할 기본키 후보 컬럼명임

    Returns:
        dict[str, Any]: 행·컬럼 수, 결측·형식·중복 지표와 참고 점수임

    Raises:
        ValueError: 입력 선택, 행·컬럼·셀·문자 수 또는 CSV 구조가 유효하지 않을 때 발생함

    Caveats:
        공식 데이터 품질 판정이 아닌 설명 가능한 학습용 휴리스틱임
    """
    rows: list[dict[str, Any]] = []

    csv_row_schema_mismatch_count = 0
    if csv_text:
        if len(csv_text) > MAX_TEXT_CHARACTERS:
            raise ValueError(f"CSV 입력이 허용 한도({MAX_TEXT_CHARACTERS}자)를 초과했습니다.")
        try:
            reader = csv.reader(io.StringIO(csv_text.strip()))
            headers = next(reader, [])
            if any(not header.strip() for header in headers):
                raise ValueError("CSV 헤더에 빈 컬럼명이 있습니다.")
            if len(headers) != len(set(headers)):
                raise ValueError("CSV 헤더에 중복 컬럼명이 있습니다.")
            if len(headers) > MAX_STRUCTURED_COLUMNS:
                raise ValueError(f"CSV 컬럼 수가 허용 한도({MAX_STRUCTURED_COLUMNS})를 초과했습니다.")

            retained_cells = 0
            rows = []
            for values in reader:
                if not values:
                    continue
                if len(rows) >= MAX_STRUCTURED_ROWS:
                    raise ValueError(f"CSV 행 수가 허용 한도({MAX_STRUCTURED_ROWS})를 초과했습니다.")
                if len(values) != len(headers):
                    csv_row_schema_mismatch_count += 1
                retained_cells += max(len(headers), len(values))
                if retained_cells > MAX_STRUCTURED_CELLS:
                    raise ValueError(f"CSV 셀 수가 허용 한도({MAX_STRUCTURED_CELLS})를 초과했습니다.")
                padded_values = values[: len(headers)] + [None] * max(0, len(headers) - len(values))
                rows.append(dict(zip(headers, padded_values)))
        except Exception as exc:
            raise ValueError(f"CSV 파싱 실패: {exc}") from exc
    elif records is not None:
        if len(records) > MAX_STRUCTURED_ROWS:
            raise ValueError(f"레코드 행 수가 허용 한도({MAX_STRUCTURED_ROWS})를 초과했습니다.")
        rows = records
        retained_cells = 0
        retained_characters = 0
        expected_column_count = len(rows[0]) if rows else 0
        for row in rows:
            if len(row) > MAX_STRUCTURED_COLUMNS:
                raise ValueError(f"레코드 컬럼 수가 허용 한도({MAX_STRUCTURED_COLUMNS})를 초과했습니다.")
            retained_cells += max(expected_column_count, len(row))
            if retained_cells > MAX_STRUCTURED_CELLS:
                raise ValueError(f"레코드 셀 수가 허용 한도({MAX_STRUCTURED_CELLS})를 초과했습니다.")
            for key, value in row.items():
                retained_characters += len(key)
                if retained_characters > MAX_TEXT_CHARACTERS:
                    raise ValueError(f"레코드 입력이 허용 한도({MAX_TEXT_CHARACTERS}자)를 초과했습니다.")
                if value is None:
                    continue
                if isinstance(value, str):
                    retained_characters += len(value)
                elif isinstance(value, (int, float, bool)):
                    retained_characters += 1
                else:
                    raise ValueError("레코드는 문자열·숫자·불리언·null 값만 포함할 수 있습니다.")
                if retained_characters > MAX_TEXT_CHARACTERS:
                    raise ValueError(f"레코드 입력이 허용 한도({MAX_TEXT_CHARACTERS}자)를 초과했습니다.")
    else:
        raise ValueError("csv_text 또는 records 중 하나를 입력해야 합니다.")

    total_rows = len(rows)
    if total_rows == 0:
        return {
            "score": 0,
            "grade": "로컬 3단계 (데이터 없음)",
            "assessment_scope": "rag-vllm-local-heuristic-not-official",
            "rule_set_version": "structured-local-v3",
            "total_rows": 0,
            "total_columns": 0,
            "columns": [],
            "row_schema_mismatch_count": 0,
            "overall_null_ratio": 0.0,
            "column_metrics": {},
            "issues": ["데이터셋이 비어 있습니다."],
        }

    columns = list(rows[0].keys())
    expected_columns = set(columns)
    row_schema_mismatch_count = csv_row_schema_mismatch_count + sum(
        1 for row in rows if set(row.keys()) != expected_columns
    )
    if not columns:
        return {
            "score": 0,
            "grade": "로컬 3단계 (컬럼 없음)",
            "assessment_scope": "rag-vllm-local-heuristic-not-official",
            "rule_set_version": "structured-local-v3",
            "total_rows": total_rows,
            "total_columns": 0,
            "columns": [],
            "row_schema_mismatch_count": row_schema_mismatch_count,
            "overall_null_ratio": 0.0,
            "column_metrics": {},
            "issues": [
                "분석 가능한 컬럼이 없습니다.",
                *([f"행 {row_schema_mismatch_count}건이 첫 행과 컬럼 구성이 다릅니다."] if row_schema_mismatch_count else []),
            ],
        }

    col_metrics: dict[str, Any] = {}
    total_null_cells = 0
    total_cells = total_rows * len(columns)
    issues: list[str] = []

    # 1. 컬럼별 결측·유일성·도메인 형식 점검함
    for col in columns:
        values = [row.get(col) for row in rows]
        # 결측 또는 공백 수 계산함
        null_count = sum(1 for v in values if v is None or str(v).strip() == "" or str(v).lower() in {"null", "none", "nan", "-"})
        total_null_cells += null_count
        null_ratio = null_count / total_rows

        # 유효값 목록과 유일값 수 계산함
        valid_vals = [str(v).strip() for v in values if v is not None and str(v).strip() not in {"", "null", "none", "nan", "-"}]
        unique_count = len(set(valid_vals))

        # 표본 기반 도메인 형식 자동 추정함
        domain_type = "일반 텍스트"
        invalid_format_count = 0

        if valid_vals:
            # 날짜·연락처·이메일·숫자형 순서로 형식 추정함
            sample = valid_vals[:20]
            if sum(1 for s in sample if DATE_PATTERN.match(s)) >= len(sample) * 0.7:
                domain_type = "날짜 (YYYY-MM-DD)"
                invalid_format_count = sum(1 for v in valid_vals if not _is_valid_date(v))
            elif sum(1 for s in sample if PHONE_PATTERN.match(s)) >= len(sample) * 0.7:
                domain_type = "연락처 (전화번호)"
                invalid_format_count = sum(1 for v in valid_vals if not PHONE_PATTERN.match(v))
            elif sum(1 for s in sample if EMAIL_PATTERN.match(s)) >= len(sample) * 0.7:
                domain_type = "이메일"
                invalid_format_count = sum(1 for v in valid_vals if not EMAIL_PATTERN.match(v))
            elif all(v.replace(".", "", 1).isdigit() for v in sample):
                domain_type = "숫자형 도메인"
                invalid_format_count = sum(1 for v in valid_vals if not v.replace(".", "", 1).isdigit())

        col_metrics[col] = {
            "null_count": null_count,
            "null_ratio": round(null_ratio, 4),
            "unique_count": unique_count,
            "detected_domain": domain_type,
            "invalid_format_count": invalid_format_count,
        }

        if null_ratio > 0.05:
            issues.append(
                f"컬럼 '{col}': 결측치 비율이 {round(null_ratio * 100, 1)}%입니다 (내부 경고선 5% 초과; 필수값 기준은 업무별 설정 필요)."
            )
        if invalid_format_count > 0:
            if domain_type.startswith("날짜"):
                issue = f"컬럼 '{col}': 유효하지 않은 날짜 또는 형식 불일치 값이 {invalid_format_count}건 있습니다."
            else:
                issue = f"컬럼 '{col}': 자동 추정 형식({domain_type})과 다른 값이 {invalid_format_count}건 있습니다. 형식 추정을 검토하십시오."
            issues.append(issue)

    if row_schema_mismatch_count:
        issues.append(
            f"행 {row_schema_mismatch_count}건에서 첫 행과 컬럼 구성이 다릅니다. 누락·추가 컬럼을 확인하십시오."
        )

    # 2. 기본키 후보 중복 여부 점검함
    pk_duplicate_count = 0
    if primary_key_col and primary_key_col in columns:
        pk_values = [str(r.get(primary_key_col, "")).strip() for r in rows if r.get(primary_key_col)]
        pk_duplicate_count = len(pk_values) - len(set(pk_values))
        if pk_duplicate_count > 0:
            issues.append(
                f"기본키 후보 컬럼 '{primary_key_col}'에 중복값 {pk_duplicate_count}건이 있습니다. 키 지정과 중복 처리 기준을 검토하십시오."
            )
    elif primary_key_col:
        issues.append(f"지정한 기본키 컬럼 '{primary_key_col}'이 데이터에 없습니다.")

    # 3. 내부 참고용 100점 점수 계산함
    completeness_score = max(0, 100 - int((total_null_cells / total_cells) * 200))  # NOTE: 0~100 범위로 제한함
    validity_penalty = sum(min(20, m["invalid_format_count"] * 2) for m in col_metrics.values())
    uniqueness_penalty = min(30, pk_duplicate_count * 5)
    schema_penalty = min(20, row_schema_mismatch_count * 2)

    final_score = max(0, min(100, completeness_score - validity_penalty - uniqueness_penalty - schema_penalty))

    if final_score >= 95:
        grade = "로컬 1단계 (참고)"
    elif final_score >= 80:
        grade = "로컬 2단계 (참고)"
    else:
        grade = "로컬 3단계 (참고)"

    return {
        "score": final_score,
        "grade": grade,
        "assessment_scope": "rag-vllm-local-heuristic-not-official",
        "rule_set_version": "structured-local-v3",
        "total_rows": total_rows,
        "total_columns": len(columns),
        "columns": columns,
        "row_schema_mismatch_count": row_schema_mismatch_count,
        "overall_null_ratio": round(total_null_cells / total_cells, 4),
        "column_metrics": col_metrics,
        "issues": issues,
    }

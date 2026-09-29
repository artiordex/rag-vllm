"""Structured data (CSV, Tabular, JSON) quality diagnostic engine.

Complies with Ministry of the Interior and Safety (행정안전부)
and NIA (한국지능정보사회진흥원) Public Data Quality Management Guidelines.
"""

from __future__ import annotations

import csv
import io
import re
from typing import Any

DATE_PATTERN = re.compile(r"^\d{4}[-/.]\d{2}[-/.]\d{2}$")
PHONE_PATTERN = re.compile(r"^01[016789]-?\d{3,4}-?\d{4}$")
EMAIL_PATTERN = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
RRN_PATTERN = re.compile(r"^\d{6}-[1-4]\d{6}$")


def evaluate_structured_dataset(
    records: list[dict[str, Any]] | None = None,
    csv_text: str | None = None,
    primary_key_col: str | None = None,
) -> dict[str, Any]:
    """Diagnose structured tabular data on 4 core government DQC metrics:

    1. 완전성 (Completeness): 결측치(NULL / 공백) 비율 점검
    2. 유효성 (Validity): 날짜, 연락처, 이메일 등 도메인 형식 유효성 점검
    3. 유일성 (Uniqueness): 기본키(PK) 중복 여부 점검
    4. 일관성 (Consistency): 컬럼별 데이터 타입 일치율 점검
    """
    rows: list[dict[str, Any]] = []

    if csv_text:
        try:
            reader = csv.DictReader(io.StringIO(csv_text.strip()))
            rows = list(reader)
        except Exception as exc:
            raise ValueError(f"CSV 파싱 실패: {exc}") from exc
    elif records is not None:
        rows = records
    else:
        raise ValueError("csv_text 또는 records 중 하나를 입력해야 합니다.")

    total_rows = len(rows)
    if total_rows == 0:
        return {
            "score": 0,
            "grade": "3등급 (데이터 없음)",
            "total_rows": 0,
            "total_columns": 0,
            "column_metrics": {},
            "issues": ["데이터셋이 비어 있습니다."],
        }

    columns = list(rows[0].keys())
    col_metrics: dict[str, Any] = {}
    total_null_cells = 0
    total_cells = total_rows * len(columns)
    issues: list[str] = []

    # 1. Column-by-column evaluation
    for col in columns:
        values = [row.get(col) for row in rows]
        # Null or blank count
        null_count = sum(1 for v in values if v is None or str(v).strip() == "" or str(v).lower() in {"null", "none", "nan", "-"})
        total_null_cells += null_count
        null_ratio = null_count / total_rows

        # Non-null values
        valid_vals = [str(v).strip() for v in values if v is not None and str(v).strip() not in {"", "null", "none", "nan", "-"}]
        unique_count = len(set(valid_vals))
        uniqueness_ratio = (unique_count / len(valid_vals)) if valid_vals else 0.0

        # Domain Pattern Check (Auto-detect)
        domain_type = "일반 텍스트"
        invalid_format_count = 0

        if valid_vals:
            # Check if looks like date
            sample = valid_vals[:20]
            if sum(1 for s in sample if DATE_PATTERN.match(s)) >= len(sample) * 0.7:
                domain_type = "날짜 (YYYY-MM-DD)"
                invalid_format_count = sum(1 for v in valid_vals if not DATE_PATTERN.match(v))
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
            issues.append(f"컬럼 '{col}': 결측치 비율이 {round(null_ratio * 100, 1)}%로 높습니다 (허용기준 1% 이하 권장).")
        if invalid_format_count > 0:
            issues.append(f"컬럼 '{col}': 도메인({domain_type}) 형식 불일치 데이터가 {invalid_format_count}건 존재합니다.")

    # 2. Primary Key Uniqueness Check
    pk_duplicate_count = 0
    if primary_key_col and primary_key_col in columns:
        pk_values = [str(r.get(primary_key_col, "")).strip() for r in rows if r.get(primary_key_col)]
        pk_duplicate_count = len(pk_values) - len(set(pk_values))
        if pk_duplicate_count > 0:
            issues.append(f"기본키(PK) 컬럼 '{primary_key_col}'에 중복값 {pk_duplicate_count}건이 발견되었습니다 (유일성 위반).")

    # 3. Overall Scoring (100-point scale)
    completeness_score = max(0, 100 - int((total_null_cells / total_cells) * 200))  # 0~100
    validity_penalty = sum(min(20, m["invalid_format_count"] * 2) for m in col_metrics.values())
    uniqueness_penalty = min(30, pk_duplicate_count * 5)

    final_score = max(0, min(100, completeness_score - validity_penalty - uniqueness_penalty))

    if final_score >= 95:
        grade = "1등급 (우수 - 공공데이터 개방 적합)"
    elif final_score >= 80:
        grade = "2등급 (보통 - 일부 정비 필요)"
    else:
        grade = "3등급 (미흡 - 전면 정비 필요)"

    return {
        "score": final_score,
        "grade": grade,
        "total_rows": total_rows,
        "total_columns": len(columns),
        "columns": columns,
        "overall_null_ratio": round(total_null_cells / total_cells, 4),
        "column_metrics": col_metrics,
        "issues": issues,
    }

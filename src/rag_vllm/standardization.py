"""Local heuristic PII candidate scanning and example term mapping.

This module is exploratory. Its regexes and built-in synonym list are not an
official standard dictionary, a compliance score, or an approval decision.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PIIMatch:
    category: str
    original: str
    masked: str
    start: int
    end: int


# =====================================================================
# 1. PII Regex Patterns (대한민국 개인정보보호법 핵심 식별자)
# =====================================================================

PII_PATTERNS = {
    "주민등록번호": re.compile(r"\b(\d{6})[- ]?([1-4]\d{6})\b"),
    "외국인등록번호": re.compile(r"\b(\d{6})[- ]?([5-8]\d{6})\b"),
    "휴대전화번호": re.compile(r"\b(01[016789])[- .]?(\d{3,4})[- .]?(\d{4})\b"),
    "일반전화번호": re.compile(r"\b(0[2-6][1-5]?)[- .]?(\d{3,4})[- .]?(\d{4})\b"),
    "이메일주소": re.compile(r"\b([a-zA-Z0-9_.+-]+)@([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)\b"),
    "신용카드번호": re.compile(r"\b(\d{4})[- ]?(\d{4})[- ]?(\d{4})[- ]?(\d{4})\b"),
    "계좌번호": re.compile(r"\b(\d{3,6})[- ](\d{2,6})[- ](\d{3,6})\b"),
}


def detect_and_mask_pii(text: str, mask: bool = True) -> tuple[str, list[dict[str, Any]]]:
    """Detect PII (Personal Identifiable Information) and optionally mask it.

    Returns:
        (processed_text, detected_issues_list)
    """
    priorities = (
        "주민등록번호",
        "외국인등록번호",
        "휴대전화번호",
        "일반전화번호",
        "이메일주소",
        "신용카드번호",
        "계좌번호",
    )

    def masked_value(category: str, match: re.Match[str]) -> str:
        groups = match.groups()
        if category in {"주민등록번호", "외국인등록번호"}:
            return f"{groups[0]}-{groups[1][0]}******"
        if category == "휴대전화번호":
            return f"{groups[0]}-****-{groups[2]}"
        if category == "일반전화번호":
            return f"{groups[0]}-****-{groups[2]}"
        if category == "이메일주소":
            user, domain = groups
            masked_user = user[:2] + "***" if len(user) > 2 else user[0] + "***"
            return f"{masked_user}@{domain}"
        if category == "신용카드번호":
            return f"{groups[0]}-****-****-{groups[3]}"
        # 계좌번호 패턴은 기관별 형식이 달라 heuristic mask로만 다룬다.
        return f"{groups[0]}-****-**{groups[2][-2:]}"

    matches: list[tuple[int, int, str, str, str]] = []
    occupied: list[tuple[int, int]] = []
    for category in priorities:
        for match in PII_PATTERNS[category].finditer(text):
            start, end = match.span()
            if any(start < used_end and end > used_start for used_start, used_end in occupied):
                continue
            occupied.append((start, end))
            matches.append((start, end, category, match.group(0), masked_value(category, match)))

    matches.sort(key=lambda item: item[0])
    detected = [
        {"category": category, "snippet": replacement, "position": start, "end": end}
        for start, end, category, _original, replacement in matches
    ]
    if not mask:
        return text, detected

    masked_text = text
    for start, end, _category, _original, replacement in reversed(matches):
        masked_text = masked_text[:start] + replacement + masked_text[end:]
    return masked_text, detected


# =====================================================================
# 2. Local example synonyms (official dictionary matching is not implemented here)
# =====================================================================

ADMINISTRATIVE_SYNONYMS = {
    # 비표준어/일상어: 행정 표준어
    "주소록": "연락처목록",
    "과태금": "과태료",
    "주민번호": "주민등록번호",
    "핸드폰": "휴대전화",
    "휴대폰": "휴대전화",
    "첨부서류": "붙임서류",
    "시행일": "시행일자",
    "접수일": "접수일자",
    "공문": "공문서",
    "기안지": "기안문",
    "거래처": "협력기관",
    "전자메일": "이메일",
}


def standardize_administrative_terms(text: str) -> tuple[str, dict[str, int]]:
    """Apply the small built-in synonym list and return replacement counts.

    Matches are local candidates, not official dictionary matches. Review them
    with the responsible domain owner before applying them as standards.

    Returns:
        (standardized_text, {replaced_term: count})
    """
    standardized = text
    replacement_counts: dict[str, int] = {}

    for non_std, std in ADMINISTRATIVE_SYNONYMS.items():
        pattern = re.compile(rf"\b{re.escape(non_std)}\b")
        matches = len(pattern.findall(standardized))
        if matches > 0:
            standardized = pattern.sub(std, standardized)
            replacement_counts[f"{non_std} -> {std}"] = matches

    return standardized, replacement_counts


# =====================================================================
# 3. Internal illustrative quality buckets (not an official DQC score)
# =====================================================================

def evaluate_enterprise_quality(
    text: str,
    source_name: str = "",
    auto_mask_pii: bool = True,
    apply_local_term_replacements: bool = True,
) -> dict[str, Any]:
    """Calculate an internal heuristic report for unstructured text.

    The five score buckets are illustrative and do not implement or certify a
    government quality standard. Regex matches and local term candidates are
    not equivalent to verified privacy clearance or approved standardization.

    Pillars:
    1. 완전성 (Completeness): 누락 및 빈 줄 비율
    2. 유효성 (Validity): 날짜 형식 및 제어문자 오염도
    3. 일관성 (Consistency): 중복 문단 비율
    4. 무결성 (Integrity): 개인정보 보호 및 비식별화 준수율
    5. 정확성 (Accuracy): 비정상 인코딩 및 기호 깨짐 여부
    """
    # Step 1: PII Scan
    masked_text, pii_issues = detect_and_mask_pii(text, mask=auto_mask_pii)

    # Step 2: Terminology Standardization Scan
    suggested_text, term_replacements = standardize_administrative_terms(masked_text)
    clean_text = suggested_text if apply_local_term_replacements else masked_text

    # Metrics
    char_count = len(clean_text)
    lines = clean_text.splitlines() if clean_text else []
    non_empty_lines = [l.strip() for l in lines if l.strip()]
    line_count = len(lines)
    non_empty_count = len(non_empty_lines)

    # Duplicate check
    from collections import Counter
    counts = Counter(non_empty_lines)
    duplicate_line_count = sum(c - 1 for c in counts.values() if c > 1)
    dup_ratio = (duplicate_line_count / non_empty_count) if non_empty_count else 0.0

    # Contaminations
    replacement_count = clean_text.count("\uFFFD")
    control_count = sum(1 for c in clean_text if ord(c) < 32 and c not in {"\n", "\t"})
    blank_ratio = ((line_count - non_empty_count) / line_count) if line_count else 0.0

    # Pillar Scores (out of 20 each, total 100)
    # 1. Completeness (완전성)
    score_completeness = 20
    if char_count < 100:
        score_completeness -= 10
    if blank_ratio > 0.4:
        score_completeness -= 5

    # 2. Validity (유효성)
    score_validity = 20
    if control_count > 0:
        score_validity -= min(10, control_count)

    # 3. Consistency (일관성)
    score_consistency = 20
    if dup_ratio >= 0.25:
        score_consistency -= 12
    elif dup_ratio >= 0.10:
        score_consistency -= 5

    # 4. Integrity (무결성 - PII)
    score_integrity = 20
    if pii_issues:
        # PII found: alert on unmasked or reward for auto-masked
        if not auto_mask_pii:
            score_integrity -= min(15, len(pii_issues) * 5)
        else:
            score_integrity -= min(5, len(pii_issues))  # slight penalty for raw PII ingestion

    # 5. Accuracy (정확성 - 인코딩 및 오염)
    score_accuracy = 20
    if replacement_count > 0:
        score_accuracy -= min(15, replacement_count * 3)

    total_score = max(0, min(100, score_completeness + score_validity + score_consistency + score_integrity + score_accuracy))

    if total_score >= 90:
        grade = "1등급 (우수)"
        status = "pass"
    elif total_score >= 70:
        grade = "2등급 (보통)"
        status = "warn"
    else:
        grade = "3등급 (미흡/개선필요)"
        status = "fail"

    recommendations: list[str] = []
    if pii_issues:
        if auto_mask_pii:
            recommendations.append(
                f"규칙 기반 개인정보 후보 {len(pii_issues)}건을 마스킹했습니다. 완전한 비식별화 여부는 별도 검토가 필요합니다."
            )
        else:
            recommendations.append(
                f"규칙 기반 개인정보 후보 {len(pii_issues)}건을 찾았습니다. 원문은 마스킹하지 않았습니다."
            )
    if term_replacements:
        action = "적용했습니다" if apply_local_term_replacements else "후보로 찾았습니다"
        recommendations.append(
            f"rag-vllm 내장 용어 목록에서 치환 후보 {len(term_replacements)}종을 {action}. 공식 표준사전 검증이나 승인은 아닙니다."
        )
    if replacement_count:
        recommendations.append(
            f"인코딩 손상 대체 문자(U+FFFD) {replacement_count}개가 감지되었습니다. 파일 저장 인코딩을 확인하십시오."
        )
    if dup_ratio >= 0.15:
        recommendations.append("중복 문단 비율이 높습니다. 본문 텍스트의 중복 추출 여부를 점검하십시오.")

    return {
        "score": total_score,
        "assessment_scope": "rag-vllm-local-heuristic-not-official",
        "grade": grade,
        "status": status,
        "pillars": {
            "완전성_Completeness": score_completeness,
            "유효성_Validity": score_validity,
            "일관성_Consistency": score_consistency,
            "무결성_Integrity": score_integrity,
            "정확성_Accuracy": score_accuracy,
        },
        "pii_detected_count": len(pii_issues),
        "pii_details": pii_issues,
        "standardized_terms_count": len(term_replacements),
        "standardized_terms": term_replacements,
        "standardization_status": "local_candidates_pending_review",
        "standardization_source": "rag-vllm_builtin_synonyms",
        "standardization_applied": apply_local_term_replacements,
        "pii_scan_scope": "heuristic_pattern_match",
        "pii_masking_applied": auto_mask_pii,
        "recommendations": recommendations,
        "cleaned_text": clean_text,
    }

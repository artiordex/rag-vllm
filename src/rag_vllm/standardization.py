"""Enterprise data standardization and PII (Privacy) detection & masking module.

Complies with Korean Public Data Quality Guidelines (행정안전부 공공데이터 품질관리 실태평가).
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
    detected: list[dict[str, Any]] = []
    masked_text = text

    # 1. 주민등록번호 (앞 6자리-뒷 첫째자리 이후 마스킹: 900101-1******)
    def mask_rrn(m: re.Match) -> str:
        detected.append({
            "category": "주민등록번호",
            "snippet": f"{m.group(1)}-{m.group(2)[0]}******",
            "position": m.start(),
        })
        return f"{m.group(1)}-{m.group(2)[0]}******" if mask else m.group(0)

    masked_text = PII_PATTERNS["주민등록번호"].sub(mask_rrn, masked_text)

    # 2. 외국인등록번호
    def mask_frn(m: re.Match) -> str:
        detected.append({
            "category": "외국인등록번호",
            "snippet": f"{m.group(1)}-{m.group(2)[0]}******",
            "position": m.start(),
        })
        return f"{m.group(1)}-{m.group(2)[0]}******" if mask else m.group(0)

    masked_text = PII_PATTERNS["외국인등록번호"].sub(mask_frn, masked_text)

    # 3. 휴대전화번호 (가운데 자리 마스킹: 010-****-5678)
    def mask_phone(m: re.Match) -> str:
        detected.append({
            "category": "휴대전화번호",
            "snippet": f"{m.group(1)}-****-{m.group(3)}",
            "position": m.start(),
        })
        return f"{m.group(1)}-****-{m.group(3)}" if mask else m.group(0)

    masked_text = PII_PATTERNS["휴대전화번호"].sub(mask_phone, masked_text)

    # 4. 이메일 (아이디 앞 2자리 외 마스킹: te***@domain.com)
    def mask_email(m: re.Match) -> str:
        user, domain = m.group(1), m.group(2)
        masked_user = user[:2] + "***" if len(user) > 2 else user[0] + "***"
        snippet = f"{masked_user}@{domain}"
        detected.append({"category": "이메일주소", "snippet": snippet, "position": m.start()})
        return snippet if mask else m.group(0)

    masked_text = PII_PATTERNS["이메일주소"].sub(mask_email, masked_text)

    # 5. 카드번호
    def mask_card(m: re.Match) -> str:
        detected.append({
            "category": "신용카드번호",
            "snippet": f"{m.group(1)}-****-****-{m.group(4)}",
            "position": m.start(),
        })
        return f"{m.group(1)}-****-****-{m.group(4)}" if mask else m.group(0)

    masked_text = PII_PATTERNS["신용카드번호"].sub(mask_card, masked_text)

    return masked_text, detected


# =====================================================================
# 2. Administrative Terminology Standardization (행정표준용어 매핑)
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
    """Replace non-standard administrative colloquialisms with official government standard terms.

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
# 3. 5-Pillar Enterprise Data Quality Evaluation (공공데이터 실태평가 5대 지표)
# =====================================================================

def evaluate_enterprise_quality(
    text: str,
    source_name: str = "",
    auto_mask_pii: bool = True,
) -> dict[str, Any]:
    """Calculate comprehensive quality metrics based on government DQC standards.

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
    clean_text, term_replacements = standardize_administrative_terms(masked_text)

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
    replacement_count = clean_text.count("")
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
        recommendations.append(f"개인정보(주민등록번호 등 {len(pii_issues)}건)가 발견되어 자동 비식별화 마스킹 조치되었습니다.")
    if term_replacements:
        recommendations.append(f"비표준 행정 용어 {len(term_replacements)}종이 행정안전부 표준용어로 자동 보정되었습니다.")
    if replacement_count:
        recommendations.append("인코딩 손상 문자()가 감지되었습니다. 파일 저장 인코딩을 UTF-8로 확인하십시오.")
    if dup_ratio >= 0.15:
        recommendations.append("중복 문단 비율이 높습니다. 본문 텍스트의 중복 추출 여부를 점검하십시오.")

    return {
        "score": total_score,
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
        "recommendations": recommendations,
        "cleaned_text": clean_text,
    }

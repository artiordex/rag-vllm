# =============================================================================
# 파일명: standardization.py
# 경로: src/rag_vllm/standardization.py
# 목적: 개인정보 후보 탐지·마스킹과 예시 용어 표준화·품질 평가 수행함
# 작성자: AI전략팀
# 작성일: 2026-09-30
# 수정일: 2026-09-30
# =============================================================================

"""개인정보 후보 탐지·마스킹과 예시 용어 표준화·품질 평가 수행함

공식 사전·품질 점수·승인 판정이 아닌 탐색용 규칙으로 동작함
"""

from __future__ import annotations

import re
from typing import Any

from .chunking import MAX_TEXT_CHARACTERS

# =====================================================================
# 1. 국내 식별자·연락처 후보를 탐지하기 위한 휴리스틱 패턴
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
    """개인정보 후보를 탐지하고 선택적으로 가림 처리함

    Args:
        text: 개인정보 후보를 탐지할 원문임
        mask: 탐지된 값을 가릴지 여부임

    Returns:
        tuple[str, list[dict[str, Any]]]: 처리 본문과 탐지 위치 목록임

    Caveats:
        정규식 기반 후보 탐지이므로 완전한 비식별화나 법적 안전 판정이 아님
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
        """탐지 유형별 개인정보 표시 규칙을 적용한 대체값을 생성함"""
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
        # NOTE: 계좌번호 형식은 기관별 차이가 있어 보수적인 부분 마스킹만 적용함
        return f"{groups[0]}-****-**{groups[2][-2:]}"

    candidates: list[tuple[int, int, int, str, str]] = []
    for priority, category in enumerate(priorities):
        for match in PII_PATTERNS[category].finditer(text):
            start, end = match.span()
            replacement = masked_value(category, match) if mask else ""
            candidates.append((start, end, priority, category, replacement))

    candidates.sort(key=lambda item: (item[0], item[2], -(item[1] - item[0])))
    matches: list[tuple[int, int, str, str]] = []
    last_end = -1
    for start, end, _priority, category, replacement in candidates:
        if start < last_end:
            continue
        matches.append((start, end, category, replacement))
        last_end = end

    detected = [
        {"category": category, "position": start, "end": end}
        for start, end, category, _replacement in matches
    ]
    if not mask:
        return text, detected

    masked_text = text
    for start, end, _category, replacement in reversed(matches):
        masked_text = masked_text[:start] + replacement + masked_text[end:]
    return masked_text, detected


# =====================================================================
# 2. 공식 사전이 아닌 로컬 예시 용어 매핑
# =====================================================================

ADMINISTRATIVE_SYNONYMS = {
    # NOTE: 공식 표준사전 항목이 아니므로 업무 적용 전 담당자 검토가 필요함
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
    """내장 예시 용어 목록을 적용하고 치환 횟수를 반환함

    로컬 후보는 공식 표준사전 일치 결과가 아니므로 표준어로 적용하기 전에
    담당 도메인 소유자의 검토가 필요함

    Returns:
        tuple[str, dict[str, int]]: 치환 본문과 원어·표준어별 횟수임
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
# 3. 공식 DQC 점수가 아닌 내부 참고용 품질 구간
# =====================================================================

def evaluate_enterprise_quality(
    text: str,
    source_name: str = "",
    auto_mask_pii: bool = True,
    apply_local_term_replacements: bool = True,
) -> dict[str, Any]:
    """비정형 텍스트에 개인정보·용어·중복 규칙을 적용해 참고 리포트를 산출함

    Args:
        text: 진단할 원문 텍스트임
        source_name: 리포트에 기록할 원천 이름임
        auto_mask_pii: 개인정보 후보를 결과 본문에서 가릴지 여부임
        apply_local_term_replacements: 로컬 예시 용어 치환을 결과 본문에 적용할지 여부임

    Returns:
        dict[str, Any]: 내부 참고 점수와 개인정보·용어 탐지 결과임

    Raises:
        ValueError: 입력 텍스트가 최대 문자 수를 초과할 때 발생함

    Caveats:
        다섯 점수 구간은 정부 품질 표준 인증이나 개인정보 완전 비식별화
        판정과 동일하지 않음
    """
    if len(text) > MAX_TEXT_CHARACTERS:
        raise ValueError(f"진단 텍스트가 허용 한도({MAX_TEXT_CHARACTERS}자)를 초과했습니다.")

    # NOTE: 개인정보 후보를 먼저 처리해 이후 품질 계산에 원문 노출을 줄임
    masked_text, pii_issues = detect_and_mask_pii(text, mask=auto_mask_pii)

    # NOTE: 표준화 후보는 공식 사전이 아니므로 설정에 따라 적용 여부를 분리함
    suggested_text, term_replacements = standardize_administrative_terms(masked_text)
    clean_text = suggested_text if apply_local_term_replacements else masked_text

    # 진단 지표 계산함
    char_count = len(clean_text)
    lines = clean_text.splitlines() if clean_text else []
    non_empty_lines = [l.strip() for l in lines if l.strip()]
    line_count = len(lines)
    non_empty_count = len(non_empty_lines)

    # 반복 문단 비율 계산함
    from collections import Counter
    counts = Counter(non_empty_lines)
    duplicate_line_count = sum(c - 1 for c in counts.values() if c > 1)
    dup_ratio = (duplicate_line_count / non_empty_count) if non_empty_count else 0.0

    # 인코딩 대체 문자와 제어문자 오염 계산함
    replacement_count = clean_text.count("\uFFFD")
    control_count = sum(1 for c in clean_text if ord(c) < 32 and c not in {"\n", "\t"})
    blank_ratio = ((line_count - non_empty_count) / line_count) if line_count else 0.0

    # NOTE: 내부 참고용 5개 축 20점씩의 100점 체계임
    # 1. 완전성 점수 계산함
    score_completeness = 20
    if char_count < 100:
        score_completeness -= 10
    if blank_ratio > 0.4:
        score_completeness -= 5

    # 2. 유효성 점수 계산함
    score_validity = 20
    if control_count > 0:
        score_validity -= min(10, control_count)

    # 3. 일관성 점수 계산함
    score_consistency = 20
    if dup_ratio >= 0.25:
        score_consistency -= 12
    elif dup_ratio >= 0.10:
        score_consistency -= 5

    # 4. 개인정보 후보 무결성 점수 계산함
    score_integrity = 20
    if pii_issues:
        # SECURITY: 마스킹을 끄면 원문 개인정보 후보가 남을 수 있어 점수를 낮춤
        if not auto_mask_pii:
            score_integrity -= min(15, len(pii_issues) * 5)
        else:
            score_integrity -= min(5, len(pii_issues))  # SECURITY: 자동 마스킹도 별도 검토 대상으로 표시함

    # 5. 정확성 점수 계산함
    score_accuracy = 20
    if replacement_count > 0:
        score_accuracy -= min(15, replacement_count * 3)

    total_score = 0 if char_count == 0 else max(
        0,
        min(100, score_completeness + score_validity + score_consistency + score_integrity + score_accuracy),
    )

    if total_score >= 90:
        grade = "로컬 1단계 (참고)"
        status = "pass"
    elif total_score >= 70:
        grade = "로컬 2단계 (참고)"
        status = "warn"
    else:
        grade = "로컬 3단계 (참고)"
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
        "rule_set_version": "enterprise-local-v1",
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
        "standardized_terms_count": sum(term_replacements.values()),
        "standardized_terms": term_replacements,
        "standardization_status": (
            "local_replacements_applied_unverified"
            if apply_local_term_replacements and term_replacements
            else "local_candidates_pending_review"
        ),
        "standardization_source": "rag-vllm_builtin_synonyms",
        "standardization_applied": bool(apply_local_term_replacements and term_replacements),
        "pii_scan_scope": "heuristic_pattern_match",
        "pii_masking_applied": bool(auto_mask_pii and pii_issues),
        "recommendations": recommendations,
        "cleaned_text": clean_text,
    }

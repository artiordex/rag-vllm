# =============================================================================
# 파일명: guardrails.py
# 경로: src/rag_vllm/guardrails.py
# 목적: RAG 입력 검증, 탈옥·프롬프트 인젝션 차단, 개인정보 보호 및 환각 검증 제공함
# 작성자: AI전략팀
# 작성일: 2026-10-01
# 수정일: 2026-10-01
# =============================================================================

"""RAG 입력 검증, 탈옥·프롬프트 인젝션 차단, 개인정보 보호 및 환각 검증 제공함"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .standardization import PII_PATTERNS, detect_and_mask_pii

logger = logging.getLogger(__name__)


class GuardrailAction(str, Enum):
    """가드레일 검증 판정 결과에 따른 처리 동작임"""

    ALLOW = "allow"
    MASK = "mask"
    FLAG = "flag"
    BLOCK = "block"


@dataclass(slots=True)
class GuardrailViolation:
    """탐지된 보안 또는 품질 위반 항목 상세임"""

    rule_name: str
    category: str
    severity: str  # "low", "medium", "high", "critical"
    message: str
    matched_pattern: str | None = None
    offset_start: int | None = None
    offset_end: int | None = None


@dataclass(slots=True)
class GuardrailResult:
    """가드레일 검사 종합 결과임"""

    passed: bool
    action: GuardrailAction
    risk_score: float  # 0.0 ~ 1.0 (높을수록 위험함)
    violations: list[GuardrailViolation] = field(default_factory=list)
    sanitized_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# 1. 프롬프트 인젝션 및 탈옥(Jailbreak) 탐지 패턴 정의
# =============================================================================

PROMPT_INJECTION_PATTERNS: list[tuple[str, str, str, re.Pattern[str]]] = [
    (
        "instruction_override",
        "prompt_injection",
        "critical",
        re.compile(
            r"(?i)(ignore\s+(all\s+)?(previous|prior|above)\s+instructions|"
            r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions|"
            r"이전\s*(지시|명령|프롬프트|규칙)(들을|들을 전부|를)?\s*(무시|폐기|잊어|취소)|"
            r"기존\s*(설정|규칙|지침)(들을|을)?\s*(무시|따르지 마))",
        ),
    ),
    (
        "system_prompt_leakage",
        "system_leak",
        "critical",
        re.compile(
            r"(?i)(repeat\s+(the\s+)?(words\s+above|system\s+prompt)|"
            r"print\s+(the\s+)?system\s+prompt|"
            r"what\s+is\s+your\s+system\s+prompt|"
            r"reveal\s+your\s+(instructions|system\s+prompt)|"
            r"시스템\s*(프롬프트|지시사항|명령어)를?\s*(출력|공개|알려줘|말해줘))",
        ),
    ),
    (
        "role_hijacking_jailbreak",
        "jailbreak",
        "high",
        re.compile(
            r"(?i)(you\s+are\s+now\s+in\s+DAN\s+mode|"
            r"do\s+anything\s+now|"
            r"unfiltered\s+mode|"
            r"developer\s+mode\s+enabled|"
            r"너는\s*지금부터\s*(무제한|자유로운|모든\s*규칙을\s*무시하는)\s*AI다|"
            r"개발자\s*모드로\s*전환)",
        ),
    ),
    (
        "delimiter_injection",
        "delimiter_attack",
        "high",
        re.compile(
            r"(```\s*(system|assistant|instruction|human|user)[\r\n]|"
            r"<\|im_start\|>|<\|im_end\|>|\[INST\]|\[/INST\]|"
            r"<\|system\|>|<\|user\|>|<\|assistant\|>)",
        ),
    ),
    (
        "hidden_instruction_comment",
        "obfuscation",
        "medium",
        re.compile(
            r"(<!--\s*(system|instruction|override|execute)[\s\S]*?-->|"
            r"/\*\s*(system|instruction|override)[\s\S]*?\*/)",
        ),
    ),
    (
        "hypothetical_jailbreak",
        "jailbreak",
        "medium",
        re.compile(
            r"(?i)(in\s+a\s+purely\s+hypothetical\s+fictional\s+scenario\s+where\s+safety\s+does\s+not\s+matter|"
            r"가상의\s*소설 속\s*상황이라고\s*가정하고\s*(법적|윤리적|보안)\s*제한\s*없이)",
        ),
    ),
]

# 위험 발화 및 유해 키워드 목록
UNSAFE_KEYWORD_PATTERNS: list[tuple[str, str, str, re.Pattern[str]]] = [
    (
        "malware_creation",
        "safety",
        "critical",
        re.compile(
            r"(?i)(악성코드\s*(제작|작성|배포)|"
            r"랜섬웨어\s*(코드|만드는 법)|"
            r"ddos\s*공격\s*(스크립트|도구)|"
            r"write\s+(a\s+)?(keylogger|ransomware|trojan|rootkit))",
        ),
    ),
    (
        "sql_injection_attempt",
        "safety",
        "high",
        re.compile(
            r"(?i)(union\s+select\s+.*\s+from|"
            r"drop\s+table\s+rag_|"
            r";\s*drop\s+database|"
            r"'\s*or\s+'1'='1)",
        ),
    ),
]


class InputGuardrail:
    """사용자 질문 및 인제스트 문서 입력의 악의적 패턴·개인정보·비정상 입력을 검증함"""

    def __init__(
        self,
        *,
        block_on_injection: bool = True,
        mask_pii: bool = True,
        max_length: int = 16384,
    ) -> None:
        self.block_on_injection = block_on_injection
        self.mask_pii = mask_pii
        self.max_length = max_length

    def validate(self, text: str) -> GuardrailResult:
        """입력 텍스트를 검증하고 판정 결과를 반환함

        Args:
            text: 검증 대상 입력 문자열임

        Returns:
            GuardrailResult: 가드레일 판정 결과 객체임
        """
        violations: list[GuardrailViolation] = []
        raw_text = text or ""

        # 1. 입력 길이 제약 검증함
        if len(raw_text) > self.max_length:
            violations.append(
                GuardrailViolation(
                    rule_name="max_length_exceeded",
                    category="input_size",
                    severity="high",
                    message=f"입력 크기가 최대 허용치({self.max_length}자)를 초과함: {len(raw_text)}자",
                )
            )

        # 2. 비가시 제어문자 및 널 바이트 탐지함
        if "\x00" in raw_text or re.search(r"[\u200B-\u200D\uFEFF]", raw_text):
            violations.append(
                GuardrailViolation(
                    rule_name="invisible_characters",
                    category="obfuscation",
                    severity="medium",
                    message="비가시 문자 또는 제어 문자가 감지됨",
                )
            )

        # 3. 프롬프트 인젝션 및 탈옥 패턴 검증함
        has_critical_injection = False
        for rule_name, category, severity, pattern in PROMPT_INJECTION_PATTERNS:
            match = pattern.search(raw_text)
            if match:
                if severity == "critical":
                    has_critical_injection = True
                violations.append(
                    GuardrailViolation(
                        rule_name=rule_name,
                        category=category,
                        severity=severity,
                        message=f"프롬프트 인젝션/탈옥 시도 감지: {rule_name}",
                        matched_pattern=match.group(0)[:60],
                        offset_start=match.start(),
                        offset_end=match.end(),
                    )
                )

        # 4. 유해/공격성 패턴 검증함
        for rule_name, category, severity, pattern in UNSAFE_KEYWORD_PATTERNS:
            match = pattern.search(raw_text)
            if match:
                if severity == "critical":
                    has_critical_injection = True
                violations.append(
                    GuardrailViolation(
                        rule_name=rule_name,
                        category=category,
                        severity=severity,
                        message=f"보안 공격 또는 유해 명령 감지: {rule_name}",
                        matched_pattern=match.group(0)[:60],
                        offset_start=match.start(),
                        offset_end=match.end(),
                    )
                )

        # 5. 개인정보(PII) 탐지 및 마스킹 처리함
        sanitized_text, pii_matches = detect_and_mask_pii(raw_text, mask=self.mask_pii)
        for pii in pii_matches:
            violations.append(
                GuardrailViolation(
                    rule_name="pii_detected",
                    category="privacy",
                    severity="medium" if pii.get("category") in {"이메일주소", "일반전화번호"} else "high",
                    message=f"개인정보 식별자 감지: {pii.get('category')}",
                    matched_pattern=f"type:{pii.get('category')}",
                    offset_start=pii.get("start"),
                    offset_end=pii.get("end"),
                )
            )

        # 위험 점수 산출함 (가중 합산 후 1.0 상한)
        risk_score = 0.0
        for v in violations:
            if v.severity == "critical":
                risk_score += 0.5
            elif v.severity == "high":
                risk_score += 0.3
            elif v.severity == "medium":
                risk_score += 0.15
            else:
                risk_score += 0.05
        risk_score = min(1.0, round(risk_score, 2))

        # 차단 / 마스킹 / 허용 판정 수행함
        security_violations = [
            v for v in violations
            if v.category in {"prompt_injection", "system_leak", "safety", "jailbreak", "input_size"}
        ]
        if has_critical_injection and self.block_on_injection:
            action = GuardrailAction.BLOCK
            passed = False
        elif any(v.severity in {"critical", "high"} for v in security_violations) and self.block_on_injection:
            action = GuardrailAction.BLOCK
            passed = False
        elif pii_matches and self.mask_pii:
            action = GuardrailAction.MASK
            passed = True
        elif violations:
            action = GuardrailAction.FLAG
            passed = True
        else:
            action = GuardrailAction.ALLOW
            passed = True

        return GuardrailResult(
            passed=passed,
            action=action,
            risk_score=risk_score,
            violations=violations,
            sanitized_text=sanitized_text,
            metadata={
                "pii_count": len(pii_matches),
                "violation_count": len(violations),
                "original_length": len(raw_text),
            },
        )


class OutputGuardrail:
    """LLM 생성 답변의 근거성(Groundedness), 환각(Hallucination), 개인정보 유출 검증함"""

    def __init__(self, *, mask_pii: bool = True) -> None:
        self.mask_pii = mask_pii

    def validate(
        self,
        answer: str | None,
        context_chunks: list[dict[str, Any]] | None = None,
    ) -> GuardrailResult:
        """생성된 답변을 검증하고 환각 점수와 유해성을 판정함

        Args:
            answer: LLM 생성 답변 문자열임
            context_chunks: 검색된 참고 청크 목록임

        Returns:
            GuardrailResult: 출력 가드레일 판정 결과임
        """
        violations: list[GuardrailViolation] = []
        if not answer:
            return GuardrailResult(
                passed=True,
                action=GuardrailAction.ALLOW,
                risk_score=0.0,
                violations=[],
                sanitized_text="",
            )

        # 1. 출력 내 개인정보(PII) 유출 탐지 및 마스킹함
        sanitized_answer, pii_matches = detect_and_mask_pii(answer, mask=self.mask_pii)
        if pii_matches:
            violations.append(
                GuardrailViolation(
                    rule_name="output_pii_leakage",
                    category="privacy",
                    severity="high",
                    message=f"답변 내 개인정보 {len(pii_matches)}건 감지됨",
                )
            )

        # 2. 유해성/시스템 프롬프트 유출 패턴 확인
        system_leak_patterns = [
            re.compile(r"(?i)you\s+are\s+a\s+helpful\s+assistant"),
            re.compile(r"(?i)my\s+instructions\s+are\s+to"),
            re.compile(r"제\s*시스템\s*프롬프트는\s*다음과\s*같습니다"),
        ]
        for pattern in system_leak_patterns:
            if pattern.search(answer):
                violations.append(
                    GuardrailViolation(
                        rule_name="system_prompt_leakage_detected",
                        category="system_leak",
                        severity="high",
                        message="답변 내 시스템 프롬프트 유출 징후 감지됨",
                    )
                )

        # 3. 근거 청크 대조 기반 환각(Faithfulness) 분석함
        hallucination_score = 0.0
        if context_chunks:
            combined_context = " ".join(
                str(c.get("content", "")) for c in context_chunks if isinstance(c, dict)
            )
            # 답변 문장 단위 분할 후 문맥 내 키워드 일치율 측정함
            sentences = [s.strip() for s in re.split(r"[.?!]\s+", answer) if len(s.strip()) > 10]
            if sentences and combined_context:
                unsupported_count = 0
                for sent in sentences:
                    # 핵심 명사/단어(3자 이상) 추출함
                    tokens = [t for t in re.findall(r"[가-힣a-zA-Z0-9]{3,}", sent)]
                    if not tokens:
                        continue
                    supported_tokens = sum(1 for t in tokens if t in combined_context)
                    overlap_ratio = supported_tokens / len(tokens)
                    if overlap_ratio < 0.25:
                        unsupported_count += 1

                hallucination_score = round(unsupported_count / max(1, len(sentences)), 2)
                if hallucination_score > 0.6:
                    violations.append(
                        GuardrailViolation(
                            rule_name="high_hallucination_risk",
                            category="faithfulness",
                            severity="high",
                            message=f"문맥 미지원 진술 비율 높음 (환각 위험 점수: {hallucination_score})",
                        )
                    )
                elif hallucination_score > 0.3:
                    violations.append(
                        GuardrailViolation(
                            rule_name="moderate_hallucination_risk",
                            category="faithfulness",
                            severity="medium",
                            message=f"문맥 미지원 진술 일부 존재 (환각 위험 점수: {hallucination_score})",
                        )
                    )

        risk_score = min(1.0, round(len(violations) * 0.25 + hallucination_score * 0.5, 2))

        if any(v.severity == "critical" for v in violations):
            action = GuardrailAction.BLOCK
            passed = False
        elif pii_matches and self.mask_pii:
            action = GuardrailAction.MASK
            passed = True
        elif violations:
            action = GuardrailAction.FLAG
            passed = True
        else:
            action = GuardrailAction.ALLOW
            passed = True

        return GuardrailResult(
            passed=passed,
            action=action,
            risk_score=risk_score,
            violations=violations,
            sanitized_text=sanitized_answer,
            metadata={
                "hallucination_score": hallucination_score,
                "pii_count": len(pii_matches),
            },
        )

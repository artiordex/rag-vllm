#!/usr/bin/env python3
"""Seed sample public/administrative documents into rag-vllm for live POC demonstration."""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_vllm.config import get_settings
from rag_vllm.parsers import parse_document
from rag_vllm.service import ingest_text


def create_sample_hwpx() -> bytes:
    """Create a realistic in-memory Korean HWPX administrative document."""
    section_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<hs:sec xmlns:hs="http://www.hancom.co.kr/hwpml/2011/section"
        xmlns:hp="http://www.hancom.co.kr/hwpml/2011/paragraph">
    <hp:p><hp:run><hp:t>식품의약품안전처 고시 제2026-14호</hp:t></hp:run></hp:p>
    <hp:p><hp:run><hp:t>문서번호: 식약처-식품안전-2026-0891</hp:t></hp:run></hp:p>
    <hp:p><hp:run><hp:t>시행일자: 2026년 3월 1일</hp:t></hp:run></hp:p>
    <hp:p><hp:run><hp:t>수신: 전국 특별시장·광역시장·특별자치시장·도지사 및 시·군·구청장</hp:t></hp:run></hp:p>
    <hp:p><hp:run><hp:t>발신: 식품의약품안전처장</hp:t></hp:run></hp:p>
    <hp:p><hp:run><hp:t>담당부서: 식품안전정책과 (담당자: 김철수 사무관, 043-719-2015)</hp:t></hp:run></hp:p>
    <hp:p><hp:run><hp:t>제목: 2026년도 식품접객업 및 제조업 지도점검 및 행정처분 운영 규정</hp:t></hp:run></hp:p>
    <hp:p><hp:run><hp:t>1. 추진 배경 및 목적</hp:t></hp:run></hp:p>
    <hp:p><hp:run><hp:t>가. 국민 다소비 식품의 위생 안전을 선제적으로 확보하고 유통기한 변조 및 비위생적 취급 행위를 근절하고자 함.</hp:t></hp:run></hp:p>
    <hp:p><hp:run><hp:t>나. 식품위생법 제44조 및 제75조에 따른 행정처분 기준의 일관성과 객관성을 확립하여 지방자치단체 행정의 신뢰도를 제고함.</hp:t></hp:run></hp:p>
    <hp:p><hp:run><hp:t>2. 위반 행위별 행정처분 및 과태료 부과 기준</hp:t></hp:run></hp:p>
    <hp:tbl>
        <hp:tr>
            <hp:tc><hp:p><hp:t>위반 행위 구분</hp:t></hp:p></hp:tc>
            <hp:tc><hp:p><hp:t>근거 법령</hp:t></hp:p></hp:tc>
            <hp:tc><hp:p><hp:t>1차 위반 처분</hp:t></hp:p></hp:tc>
            <hp:tc><hp:p><hp:t>2차 위반 처분</hp:t></hp:p></hp:tc>
            <hp:tc><hp:p><hp:t>3차 위반 처분</hp:t></hp:p></hp:tc>
        </hp:tr>
        <hp:tr>
            <hp:tc><hp:p><hp:t>소비기한(유통기한) 경과제품 조리·판매 목적 보관</hp:t></hp:p></hp:tc>
            <hp:tc><hp:p><hp:t>식품위생법 제44조제1항</hp:t></hp:p></hp:tc>
            <hp:tc><hp:p><hp:t>영업정지 7일</hp:t></hp:p></hp:tc>
            <hp:tc><hp:p><hp:t>영업정지 15일</hp:t></hp:p></hp:tc>
            <hp:tc><hp:p><hp:t>영업정지 1개월</hp:t></hp:p></hp:tc>
        </hp:tr>
        <hp:tr>
            <hp:tc><hp:p><hp:t>조리장 시설기준 및 위생모 미착용 등 위생관리 불량</hp:t></hp:p></hp:tc>
            <hp:tc><hp:p><hp:t>식품위생법 제36조</hp:t></hp:p></hp:tc>
            <hp:tc><hp:p><hp:t>시정명령</hp:t></hp:p></hp:tc>
            <hp:tc><hp:p><hp:t>영업정지 5일</hp:t></hp:p></hp:tc>
            <hp:tc><hp:p><hp:t>영업정지 10일</hp:t></hp:p></hp:tc>
        </hp:tr>
        <hp:tr>
            <hp:tc><hp:p><hp:t>원산지 거짓 표시 또는 혼동 유발 행위</hp:t></hp:p></hp:tc>
            <hp:tc><hp:p><hp:t>농수산물의 원산지표시법 제6조</hp:t></hp:p></hp:tc>
            <hp:tc><hp:p><hp:t>영업정지 15일</hp:t></hp:p></hp:tc>
            <hp:tc><hp:p><hp:t>영업정지 1개월</hp:t></hp:p></hp:tc>
            <hp:tc><hp:p><hp:t>영업허가 취소</hp:t></hp:p></hp:tc>
        </hp:tr>
        <hp:tr>
            <hp:tc><hp:p><hp:t>건강진단 미실시(보건증 미구비) 종사자 고용</hp:t></hp:p></hp:tc>
            <hp:tc><hp:p><hp:t>식품위생법 제40조</hp:t></hp:p></hp:tc>
            <hp:tc><hp:p><hp:t>과태료 20만원</hp:t></hp:p></hp:tc>
            <hp:tc><hp:p><hp:t>과태료 40만원</hp:t></hp:p></hp:tc>
            <hp:tc><hp:p><hp:t>과태료 60만원</hp:t></hp:p></hp:tc>
        </hp:tr>
    </hp:tbl>
    <hp:p><hp:run><hp:t>3. 행정처분 이행 절차 및 의견제출</hp:t></hp:run></hp:p>
    <hp:p><hp:run><hp:t>가. 행정절차법 제21조에 따라 행정처분 사전통지 시 처분 당사자에게 최소 10일 이상의 의견제출 기한을 부여하여야 함.</hp:t></hp:run></hp:p>
    <hp:p><hp:run><hp:t>나. 천재지변 등 불가피한 사유가 인정되는 경우 2분의 1 범위 내에서 처분을 경감할 수 있음.</hp:t></hp:run></hp:p>
</hs:sec>
"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("Contents/section0.xml", section_xml.encode("utf-8"))
    return buf.getvalue()


SAMPLE_EDUCATION_TXT = """[교육부 공공데이터 품질관리 및 실태평가 종합 시행계획]

1. 기본 개요
- 사업명: 2026년도 교육부 및 산하기관 공공데이터 품질관리 실태점검 및 표준화 고도화 사업
- 총사업예산: 2억 4,000만원 (부가가치세 포함)
- 사업기간: 2026.04.01 ~ 2026.11.30 (8개월간)
- 주관부서: 교육부 디지털교육정보국 디지털데이터담당관 (담당: 이영희 사무관)
- 점검대상: 교육부 본부 및 소속기관 18개 교육청, 산하 공공기관 32개소 (총 51개 데이터베이스)

2. 추진 배경 및 목적
가. 행정안전부 공공데이터 제공 및 데이터 기반 행정 평가 지침 준수
나. 국가 데이터베이스의 데이터 값 오류를 최소화하고, 공공·민간 데이터 연계 활용도를 95% 이상으로 제고함.

3. 3대 중점 품질관리 평가지표
(1) 데이터 값 정밀도 지표 (배점: 40점)
- 날짜 유효성, 주민등록번호 등 식별자 포맷 정밀도, 코드 도메인 일치율 점검
- 허용 오류율: 전체 컬럼 건수 대비 0.01% 이하 달성 의무화
(2) 메타데이터 표준 준수율 (배점: 30점)
- 범정부 메타데이터 공통 표준단어, 표준용어 및 도메인 준수 비율
- 미준수 표준항목 발견 시 14일 이내 표준화 사전 신청 등록 필요
(3) 개방 데이터 최신성 및 개방 주기 준수 (배점: 30점)
- 수기 갱신 지양, API 기반 자동 연계 배치 파이프라인 구축 권장

4. 단계별 세부 추진일정
- 2026년 4월: 사업 착수 및 지침 설명회 개최, 사전 진단 스크립트 배포
- 2026년 5월 ~ 7월: 소속 산하기관 1차 품질진단 및 오류 정비
- 2026년 8월 ~ 9월: 행정안전부 합동 현장 실태점검 (표본 20% 무작위 검증)
- 2026년 10월 ~ 11월: 최종 개선보고서 취합, 우수기관 포상 및 최종 성과보고회 개최

5. 기대효과
- 데이터 품질 오류율 0.05% -> 0.008%로 84% 대폭 개선
- 범정부 데이터 실태평가에서 전년 대비 '우수(1등급)' 등급 달성
"""


SAMPLE_SECURITY_GUIDELINE_TXT = """[과학기술정보통신부 가이드라인] 생성형 AI 공공업무 도입 및 보안 수칙 가이드라인 (2026년 개정판)

문서등록번호: 과기정통부-인공지능기반-2026-0331
시행일자: 2026-02-15
발신기관: 과학기술정보통신부 인공지능기반정책관
수신기관: 중앙행정기관, 지방자치단체, 공공기관 감사실 및 정보화담당관

1. 총칙 및 제정 목적
본 가이드라인은 공공부문에서 대규모 언어 모델(LLM) 및 검색 증강 생성(RAG) 기술을 문서 작성, 법령 질의, 대민 서비스에 도입할 때 발생할 수 있는 데이터 유출 및 정보 왜곡(환각 현상)을 예방하는 것을 목적으로 한다.

2. 공공기관 생성형 AI 4대 기본 보안 수칙
가. [망분리 및 온프레미스 배포 원칙]
공공 내부 결재문서, 대외비, 개인정보가 포함된 문서를 처리하는 RAG 시스템은 반드시 외부 상용 API(퍼블릭 클라우드)가 아닌 기관 내부 온프레미스 GPU 서버(vLLM, Ollama 등 로컬 LLM)에 폐쇄망 또는 전용 VPN으로 구축하여야 함.
나. [개인정보 및 비식별화 선조치 의무]
문서 인제스트(Ingestion) 및 프롬프트 입력 단계에서 주민등록번호, 연락처, 계좌번호 등 고유식별정보를 정규식 또는 마스킹 필터를 통해 사전에 완전히 비식별화하여야 함.
다. [산출물 인간 검수(Human-in-the-Loop) 의무]
AI가 기안한 공문서 초안이나 보고서는 최종 발송 또는 결재 상신 전 반드시 담당 공무원의 사실 확인(Fact Check)과 인용 근거 대조를 필수로 거쳐야 함.
라. [할루시네이션(환각) 방지를 위한 근거 명시]
모든 AI 답변에는 추출된 사내 원문 청크 번호, 유사도 점수, 출처 문서명을 100% 명기하여 추적 가능성을 확보하여야 함.

3. RAG 파이프라인 보안 표준 규격
- 임베딩 모델: BAAI/bge-m3 등 신뢰성 검증된 오픈소스 다국어 모델 사용 권장.
- 벡터 데이터베이스: PostgreSQL pgvector 등 내부 데이터베이스에 안전하게 격리 저장.
- 질의 로그 보관: 비인가 접근 탐지를 위해 모든 RAG 질의 내역과 IP 로그는 최소 1년간 감사용으로 보관.
"""


def main() -> None:
    settings = get_settings()
    data_dir = Path(__file__).resolve().parent.parent / "data" / "samples"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=== [RAG-vLLM 사내 POC 샘플 문서 시딩 시작] ===")

    # 1. HWPX Notice
    hwpx_path = data_dir / "식품의약품안전처_식품위생_처분고시_제2026-14호.hwpx"
    hwpx_bytes = create_sample_hwpx()
    hwpx_path.write_bytes(hwpx_bytes)
    print(f"1. HWPX 생성 완료: {hwpx_path.name} ({len(hwpx_bytes)} bytes)")

    parsed_hwpx = parse_document(hwpx_path.name, hwpx_bytes, "application/octet-stream")
    res1 = ingest_text(
        settings,
        name=hwpx_path.name,
        text=parsed_hwpx.text,
        source_type="file",
        mime_type=parsed_hwpx.mime_type,
        metadata={"category": "고시/행정처분", "organization": "식품의약품안전처", "year": 2026},
    )
    print(f"   -> 인제스트 성공! Document ID: {res1['document_id']}, 청크: {res1['chunk_count']}개, 품질: {res1['quality']['score']}점")

    # 2. Education Plan TXT
    edu_path = data_dir / "교육부_2026년도_공공데이터_품질관리_지침_및_실태평가_계획서.txt"
    edu_path.write_text(SAMPLE_EDUCATION_TXT, encoding="utf-8")
    print(f"2. 텍스트 문서 생성 완료: {edu_path.name}")

    res2 = ingest_text(
        settings,
        name=edu_path.name,
        text=SAMPLE_EDUCATION_TXT,
        source_type="file",
        mime_type="text/plain",
        metadata={"category": "사업계획/지침", "organization": "교육부", "year": 2026},
    )
    print(f"   -> 인제스트 성공! Document ID: {res2['document_id']}, 청크: {res2['chunk_count']}개, 품질: {res2['quality']['score']}점")

    # 3. Security Guidelines TXT
    sec_path = data_dir / "과기정통부_생성형AI_공공업무_도입_및_보안_가이드라인.txt"
    sec_path.write_text(SAMPLE_SECURITY_GUIDELINE_TXT, encoding="utf-8")
    print(f"3. 텍스트 문서 생성 완료: {sec_path.name}")

    res3 = ingest_text(
        settings,
        name=sec_path.name,
        text=SAMPLE_SECURITY_GUIDELINE_TXT,
        source_type="file",
        mime_type="text/plain",
        metadata={"category": "보안/가이드라인", "organization": "과학기술정보통신부", "year": 2026},
    )
    print(f"   -> 인제스트 성공! Document ID: {res3['document_id']}, 청크: {res3['chunk_count']}개, 품질: {res3['quality']['score']}점")

    print("\n=== [시딩 완료: 모든 문서가 pgvector에 성공적으로 적재되었습니다] ===")
    print("웹 브라우저에서 'http://127.0.0.1:11020/'에 접속하여 실시간 POC 대시보드를 확인하세요.")


if __name__ == "__main__":
    main()

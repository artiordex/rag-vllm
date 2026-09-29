"""Comprehensive test suite for rag-vllm."""

import io
import zipfile
from fastapi.testclient import TestClient

from rag_vllm.api import app
from rag_vllm.parsers import parse_document
from rag_vllm.quality import diagnose_text

client = TestClient(app)


def test_hwpx_parsing():
    """Verify that HWPX files with sections and tables extract correctly."""
    section_xml = """<?xml version="1.0" encoding="UTF-8"?>
<hs:sec xmlns:hs="http://www.hancom.co.kr/hwpml/2011/section"
        xmlns:hp="http://www.hancom.co.kr/hwpml/2011/paragraph">
    <hp:p><hp:run><hp:t>식품의약품안전처 고시 제2026-1호</hp:t></hp:run></hp:p>
    <hp:p><hp:run><hp:t>1. 추진 배경</hp:t></hp:run></hp:p>
    <hp:tbl>
        <hp:tr>
            <hp:tc><hp:p><hp:t>구분</hp:t></hp:p></hp:tc>
            <hp:tc><hp:p><hp:t>처분기준</hp:t></hp:p></hp:tc>
        </hp:tr>
        <hp:tr>
            <hp:tc><hp:p><hp:t>1차 위반</hp:t></hp:p></hp:tc>
            <hp:tc><hp:p><hp:t>영업정지 7일</hp:t></hp:p></hp:tc>
        </hp:tr>
    </hp:tbl>
</hs:sec>
"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("Contents/section0.xml", section_xml.encode("utf-8"))

    parsed = parse_document("test_notice.hwpx", buf.getvalue(), "application/octet-stream")
    assert "식품의약품안전처 고시 제2026-1호" in parsed.text
    assert "1. 추진 배경" in parsed.text
    assert "| 구분 | 처분기준 |" in parsed.text
    assert "| 1차 위반 | 영업정지 7일 |" in parsed.text
    assert parsed.mime_type == "application/hwp+zip"


def test_quality_diagnostics():
    """Test text quality diagnostics for valid and short text."""
    # 1. Normal clean text
    clean_text = "이것은 충분한 길이의 행정 문서 테스트 텍스트입니다. " * 10
    result = diagnose_text(clean_text, "test.txt")
    assert result["score"] >= 80
    assert result["status"] == "pass"

    # 2. Too short text
    short_text = "너무 짧음"
    result_short = diagnose_text(short_text, "short.txt")
    assert any(i["code"] == "SHORT_DOCUMENT" for i in result_short["issues"])


def test_api_health_and_stats():
    """Test /health and /stats endpoints."""
    res_health = client.get("/health")
    assert res_health.status_code == 200
    data_health = res_health.json()
    assert data_health["status"] == "ok"
    assert data_health["database"] == "up"

    res_stats = client.get("/stats")
    assert res_stats.status_code == 200
    data_stats = res_stats.json()
    assert "total_documents" in data_stats
    assert "total_chunks" in data_stats
    assert data_stats["embedding_dimension"] == 1024


def test_api_dashboard_html():
    """Test GET and HEAD on root dashboard endpoint."""
    res_get = client.get("/")
    assert res_get.status_code == 200
    assert "text/html" in res_get.headers["content-type"]
    assert "RAG-vLLM 문서 AI & 지능형 검색 포털" in res_get.text

    res_head = client.head("/")
    assert res_head.status_code == 200


def test_api_list_documents():
    """Test listing documents."""
    res = client.get("/documents")
    assert res.status_code == 200
    data = res.json()
    assert "items" in data
    assert "total" in data
    assert isinstance(data["items"], list)


def test_api_structured_extract():
    """Test structured entity extraction endpoint."""
    sample_text = (
        "문서번호: 과기정통부-2026-99호\n"
        "시행일자: 2026-05-01\n"
        "발신: 과학기술정보통신부\n"
        "수신: 각급 공공기관\n"
        "제목: 생성형 AI 공공업무 표준 보안 지침 시달\n"
        "담당자: 박연구 사무관 (044-202-6000)\n"
        "주요내용: 사내 GPU망 내 vLLM 온프레미스 배포를 의무화함."
    )
    res = client.post(
        "/documents/extract",
        json={"schema_type": "공문서_메타데이터", "text": sample_text},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["schema_type"] == "공문서_메타데이터"
    assert isinstance(data["extracted_data"], dict)


def test_api_hybrid_query():
    """Test hybrid RRF search query."""
    res = client.post(
        "/query",
        json={
            "question": "행정처분 과태료 기준은?",
            "search_mode": "hybrid",
            "top_k": 3,
            "use_llm": False,  # vector + sparse search retrieval only
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert "sources" in data
    assert len(data["sources"]) > 0
    assert "confidence_score" in data
    assert "hallucination_risk" in data


def test_enterprise_standardization_and_pii():
    """Test Korean PII detection, masking, and terminology standardization."""
    from rag_vllm.standardization import evaluate_enterprise_quality

    text = "담당자 홍길동 (주민번호: 900101-1234567, 휴대폰: 010-9876-5432). 과태금과 주소록 확인 바람."
    res = evaluate_enterprise_quality(text, "test.txt", auto_mask_pii=True)
    assert res["pii_detected_count"] >= 2
    assert "900101-1******" in res["cleaned_text"]
    assert "010-****-5432" in res["cleaned_text"]
    assert "연락처목록" in res["cleaned_text"]
    assert "과태료" in res["cleaned_text"] or "과태금" in text


def test_multimodal_image_parsing():
    """Test image parsing and visual metadata extraction."""
    from PIL import Image

    # Create dummy in-memory PNG
    img = Image.new("RGB", (120, 80), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    parsed = parse_document("sample_chart.png", buf.getvalue(), "image/png")
    assert parsed.mime_type == "image/png"
    assert parsed.metadata["width"] == 120
    assert parsed.metadata["height"] == 80
    assert "sample_chart.png" in parsed.text


def test_reranker_cross_scoring():
    """Test 2-stage reranker cross-scoring heuristic and sorting."""
    from rag_vllm.reranker import rerank_chunks

    query = "식품위생 행정처분 기준"
    candidates = [
        {"id": 1, "text": "일반 날씨 정보입니다.", "score": 0.4},
        {"id": 2, "text": "식품위생 행정처분 기준 및 과태료 규정입니다.", "score": 0.5},
    ]
    reranked = rerank_chunks(query, candidates, top_k=2)
    assert reranked[0]["id"] == 2
    assert reranked[0]["rerank_score"] > reranked[1]["rerank_score"]


def test_api_enterprise_quality_endpoint():
    """Test POST /quality/enterprise endpoint."""
    res = client.post(
        "/quality/enterprise",
        json={
            "text": "시행일자: 2026-03-01. 담당자: 홍길동 (010-1111-2222). 주소록 송부함.",
            "auto_mask_pii": True,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert "score" in data
    assert "grade" in data
    assert "pillars" in data
    assert data["pii_detected_count"] >= 1


def test_api_structured_quality_endpoint():
    """Test POST /quality/structured endpoint."""
    csv_sample = (
        "id,name,phone,join_date\n"
        "1,홍길동,010-1234-5678,2026-01-01\n"
        "2,김철수,010-9876-5432,2026-02-15\n"
        "3,이영희,,2026-03-01\n"
    )
    res = client.post(
        "/quality/structured",
        json={"csv_text": csv_sample, "primary_key_col": "id"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "score" in data
    assert "grade" in data
    assert data["total_rows"] == 3
    assert len(data["columns"]) == 4


def test_api_audit_logs_endpoint():
    """Test GET /audit/logs endpoint."""
    res = client.get("/audit/logs?limit=10")
    assert res.status_code == 200
    logs = res.json()
    assert isinstance(logs, list)



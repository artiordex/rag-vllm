"""검색 근거를 현재 RAG 문서 row의 색인 본문 digest에 결속함."""

from types import SimpleNamespace
from uuid import UUID

from rag_vllm import service


def test_pgvector_candidates_include_active_document_digest(monkeypatch) -> None:
    document_id = UUID("00000000-0000-4000-8000-000000000001")
    digest = "a" * 64
    hit = {
        "document_id": document_id,
        "source_name": "stale-vector-name.pdf",
        "text": "synthetic excerpt",
        "metadata": {"chunk_index": 0},
        "score": 0.8,
        "quality_score": 100,
    }

    def search_chunks_hybrid(*args, **kwargs):
        return [hit]

    def find_bindings(settings, document_ids, *, project_name):
        assert document_ids == [document_id]
        assert project_name == "moe_dq_diag"
        return {document_id: ("current-source.jsonl", digest)}

    monkeypatch.setattr(service, "search_chunks_hybrid", search_chunks_hybrid)
    monkeypatch.setattr(service, "find_document_source_bindings_by_ids", find_bindings)

    results = service._retrieve_candidates(
        SimpleNamespace(),
        vector_store_type="pgvector",
        query_embedding=[0.0],
        query_text="synthetic query",
        top_k=3,
        document_id=None,
        document_ids=None,
        min_quality_score=None,
        project_name="moe_dq_diag",
        department=None,
        max_security_level=None,
        search_mode="hybrid",
    )

    assert len(results) == 1
    assert results[0]["source_name"] == "current-source.jsonl"
    assert results[0]["text"] == "synthetic excerpt"
    assert results[0]["score"] == 0.8
    assert results[0]["metadata"]["project_name"] == "moe_dq_diag"
    assert results[0]["metadata"]["document_content_sha256"] == digest


def test_pgvector_candidates_drop_hits_without_an_active_document_binding(monkeypatch) -> None:
    document_id = UUID("00000000-0000-4000-8000-000000000002")
    hit = {
        "document_id": document_id,
        "source_name": "orphan-source.pdf",
        "text": "synthetic excerpt",
        "metadata": {},
        "score": 0.5,
        "quality_score": 100,
    }
    monkeypatch.setattr(service, "search_chunks", lambda *args, **kwargs: [hit])
    monkeypatch.setattr(service, "find_document_source_bindings_by_ids", lambda *args, **kwargs: {})

    results = service._retrieve_candidates(
        SimpleNamespace(),
        vector_store_type="pgvector",
        query_embedding=[0.0],
        query_text="synthetic query",
        top_k=3,
        document_id=None,
        document_ids=None,
        min_quality_score=None,
        project_name="moe_dq_diag",
        department=None,
        max_security_level=None,
        search_mode="dense",
    )

    assert results == []

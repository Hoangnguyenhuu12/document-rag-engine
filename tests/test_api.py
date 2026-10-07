"""
Unit tests for FastAPI REST endpoints in api/main.py.
"""

from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import pytest

from api.main import app
from src.schemas.rag_contracts import Citation, RAGQueryResponse

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"
    assert resp.json()["service"] == "document-rag-engine"


def test_ui_endpoint():
    resp = client.get("/ui")
    assert resp.status_code == 200
    assert "Document RAG Engine" in resp.text



def test_index_ocr_endpoint():
    mock_pipeline = MagicMock()
    mock_pipeline.index_ocr_json.return_value = {
        "status": "success",
        "doc_id": "doc_test_123",
        "file_name": "scan.pdf",
        "total_pages": 1,
        "chunks_indexed": 3
    }

    with patch("api.main.get_pipeline", return_value=mock_pipeline):
        payload = {
            "ocr_data": {
                "document_name": "scan.pdf",
                "pages": [{"page_number": 1, "text": "Nội dung scan..."}]
            },
            "source_path": "scan.pdf"
        }
        resp = client.post("/rag/index-ocr", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["chunks_indexed"] == 3


def test_query_endpoint():
    mock_pipeline = MagicMock()
    mock_resp = RAGQueryResponse(
        answer="Câu trả lời từ RAG",
        citations=[
            Citation(
                doc_name="tai_lieu.pdf",
                page_number=1,
                location_label="Trang 1",
                text_snippet="Trích đoạn tài liệu",
                score=0.92
            )
        ],
        mode="fast",
        model_used="mock-llm",
        latency_seconds=0.25
    )
    mock_pipeline.query.return_value = mock_resp

    with patch("api.main.get_pipeline", return_value=mock_pipeline):
        payload = {
            "query": "Hỏi câu gì đó?",
            "top_k": 2,
            "mode": "fast"
        }
        resp = client.post("/rag/query", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "Câu trả lời từ RAG"
        assert len(data["citations"]) == 1
        assert data["citations"][0]["doc_name"] == "tai_lieu.pdf"


def test_stream_endpoint():
    mock_pipeline = MagicMock()
    sample_citation = Citation(
        doc_name="huong_dan.pdf",
        page_number=1,
        location_label="Trang 1",
        text_snippet="Đoạn trích",
        score=0.88
    )

    def mock_generator():
        yield [sample_citation]
        yield "Từ "
        yield "khóa "
        yield "mới."

    mock_pipeline.stream_query.return_value = mock_generator()

    with patch("api.main.get_pipeline", return_value=mock_pipeline):
        payload = {
            "query": "Stream test",
            "top_k": 1
        }
        resp = client.post("/rag/stream", json=payload)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        body = resp.text
        assert "event: citations" in body
        assert "huong_dan.pdf" in body
        assert "event: token" in body
        assert "event: done" in body


def test_list_documents_endpoint():
    mock_pipeline = MagicMock()
    mock_pipeline.list_documents.return_value = [
        {"file_name": "chinh_sach.pdf", "chunk_count": 5},
        {"file_name": "bao_cao.docx", "chunk_count": 8}
    ]

    with patch("api.main.get_pipeline", return_value=mock_pipeline):
        resp = client.get("/rag/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_documents"] == 2
        assert data["documents"][0]["file_name"] == "chinh_sach.pdf"


def test_delete_document_endpoint():
    mock_pipeline = MagicMock()
    mock_pipeline.delete_document.return_value = True

    with patch("api.main.get_pipeline", return_value=mock_pipeline):
        resp = client.delete("/rag/documents/chinh_sach.pdf")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        mock_pipeline.delete_document.assert_called_once_with("chinh_sach.pdf")

    # Test not found
    mock_pipeline.delete_document.return_value = False
    with patch("api.main.get_pipeline", return_value=mock_pipeline):
        resp = client.delete("/rag/documents/khong_ton_tai.pdf")
        assert resp.status_code == 404


def test_clear_documents_endpoint():
    mock_pipeline = MagicMock()
    mock_pipeline.clear_all_documents.return_value = True

    with patch("api.main.get_pipeline", return_value=mock_pipeline):
        resp = client.delete("/rag/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        mock_pipeline.clear_all_documents.assert_called_once()


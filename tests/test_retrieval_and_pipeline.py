"""
Unit tests for VectorStoreManager, HybridRetriever, and DocumentRAGPipeline with Mock Providers.
"""

from typing import List, Generator, Optional
import pytest

from src.schemas.rag_contracts import IngestedDocument, IngestedPage, RAGQueryRequest
from src.chunking.chunker import StructureAwareChunker
from src.embeddings.embedding_provider import BaseEmbeddingProvider
from src.generation.llm_provider import BaseLLMProvider
from src.retrieval.vector_store import VectorStoreManager
from src.retrieval.retriever import HybridRetriever, OkapiBM25Scorer
from src.pipeline.rag_pipeline import DocumentRAGPipeline





def test_full_pipeline_indexing_and_query(rag_pipeline):
    # 1. Create sample IngestedDocument
    doc = IngestedDocument(
        doc_id="test_doc_01",
        file_name="chinh_sach.pdf",
        total_pages=1,
        pages=[
            IngestedPage(
                page_number=1,
                text="# Chính Sách Nhân Sự\nNhân viên được làm việc từ xa 2 ngày một tuần.",
                location_label="Trang 1"
            )
        ]
    )

    # 2. Index
    num_chunks = rag_pipeline.index_document(doc)
    assert num_chunks > 0

    # 3. Query Fast Mode
    req = RAGQueryRequest(query="Làm việc từ xa bao nhiêu ngày?")
    resp = rag_pipeline.query(req)

    assert resp.answer == "Câu trả lời thử nghiệm từ Mock LLM dựa trên tài liệu."
    assert len(resp.citations) > 0
    assert resp.citations[0].doc_name == "chinh_sach.pdf"
    assert resp.citations[0].page_number == 1
    assert "Chính Sách Nhân Sự" in resp.citations[0].text_snippet
    assert "ZERO HALLUCINATION" in rag_pipeline.llm_provider.last_system_prompt


def test_query_thinking_mode_and_history(rag_pipeline):
    doc = IngestedDocument(
        doc_id="test_doc_03",
        file_name="quy_dinh.pdf",
        total_pages=1,
        pages=[
            IngestedPage(
                page_number=1,
                text="# Quy Định Nghỉ Phép\nNhân viên có 12 ngày phép năm.",
                location_label="Trang 1"
            )
        ]
    )
    rag_pipeline.index_document(doc)

    req = RAGQueryRequest(
        query="Nếu nghỉ quá 12 ngày thì tính sao?",
        mode="thinking",
        conversation_history=[
            {"role": "user", "content": "Một năm có bao nhiêu ngày phép?"},
            {"role": "assistant", "content": "Có 12 ngày phép."}
        ]
    )
    resp = rag_pipeline.query(req)

    assert resp.mode == "thinking"
    assert "CHAIN-OF-THOUGHT" in rag_pipeline.llm_provider.last_system_prompt
    assert "LỊCH SỬ HỘI THOẠI TRƯỚC ĐÓ:" in rag_pipeline.llm_provider.last_prompt
    assert "Một năm có bao nhiêu ngày phép?" in rag_pipeline.llm_provider.last_prompt


def test_pipeline_index_ocr_json_helper(rag_pipeline):
    sample_ocr = {
        "document_name": "hop_dong_scan.pdf",
        "total_pages": 1,
        "pages": [
            {
                "page_number": 1,
                "text": "Điều khoản bảo mật thông tin khách hàng.",
                "tables": []
            }
        ]
    }
    result = rag_pipeline.index_ocr_json(sample_ocr)
    assert result["status"] == "success"
    assert result["file_name"] == "hop_dong_scan.pdf"
    assert result["chunks_indexed"] > 0


def test_streaming_query(rag_pipeline):
    doc = IngestedDocument(
        doc_id="test_doc_02",
        file_name="huong_dan.pdf",
        total_pages=1,
        pages=[
            IngestedPage(
                page_number=1,
                text="Hướng dẫn cài đặt hệ thống.",
                location_label="Trang 1"
            )
        ]
    )
    rag_pipeline.index_document(doc)

    generator = rag_pipeline.stream_query(RAGQueryRequest(query="cài đặt"))
    # First item yielded is citations list
    citations = next(generator)
    assert isinstance(citations, list)

    # Subsequent items are token strings
    tokens = list(generator)
    assert "".join(tokens) == "Câu trả lời thử nghiệm."


def test_pipeline_document_management(rag_pipeline):
    doc_a = IngestedDocument(
        doc_id="doc_crud_1",
        file_name="chinh_sach_a.pdf",
        total_pages=1,
        pages=[IngestedPage(page_number=1, text="Nội dung chính sách A", location_label="Trang 1")]
    )
    doc_b = IngestedDocument(
        doc_id="doc_crud_2",
        file_name="chinh_sach_b.pdf",
        total_pages=1,
        pages=[IngestedPage(page_number=1, text="Nội dung chính sách B", location_label="Trang 1")]
    )

    rag_pipeline.index_document(doc_a)
    rag_pipeline.index_document(doc_b)

    # 1. Test listing
    docs = rag_pipeline.list_documents()
    doc_names = [d["file_name"] for d in docs]
    assert "chinh_sach_a.pdf" in doc_names
    assert "chinh_sach_b.pdf" in doc_names

    # 2. Test delete single document
    del_res = rag_pipeline.delete_document("chinh_sach_a.pdf")
    assert del_res is True
    docs_after = rag_pipeline.list_documents()
    doc_names_after = [d["file_name"] for d in docs_after]
    assert "chinh_sach_a.pdf" not in doc_names_after
    assert "chinh_sach_b.pdf" in doc_names_after

    # 3. Test clear all documents
    clear_res = rag_pipeline.clear_all_documents()
    assert clear_res is True
    assert len(rag_pipeline.list_documents()) == 0


def test_okapi_bm25_scorer():
    scorer = OkapiBM25Scorer(k1=1.5, b=0.75)
    corpus = [
        "Quy chế nghỉ phép năm của nhân viên là 12 ngày.",
        "Hướng dẫn cài đặt môi trường lập trình Python và Docker.",
        "Chế độ công tác phí và định mức thanh toán khi đi công tác nước ngoài."
    ]
    metas = [
        {"location_label": "Trang 1 - Nghỉ phép"},
        {"location_label": "Trang 2 - Kỹ thuật"},
        {"location_label": "Trang 3 - Tài chính"}
    ]

    # Query matching doc 0
    scores = scorer.score_corpus("nghỉ phép nhân viên", corpus, metas)
    assert len(scores) == 3
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]
    assert 0.0 <= scores[0] <= 1.0

    # Query with empty result
    empty_scores = scorer.score_corpus("từ_khóa_hoàn_toàn_không_có", corpus, metas)
    assert all(s == 0.0 for s in empty_scores)




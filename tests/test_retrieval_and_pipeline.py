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
from src.retrieval.retriever import HybridRetriever
from src.pipeline.rag_pipeline import DocumentRAGPipeline


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic mock embedding for fast offline unit tests."""

    @property
    def dimension(self) -> int:
        return 4

    @property
    def model_name(self) -> str:
        return "mock-embed"

    def embed_query(self, text: str) -> List[float]:
        # Hash text to 4 floats
        h = sum(ord(c) for c in text)
        return [float((h * (i + 1)) % 10) / 10.0 for i in range(4)]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(t) for t in texts]


class MockLLMProvider(BaseLLMProvider):
    """Deterministic mock LLM for testing pipeline answers."""

    def __init__(self):
        self.last_prompt = ""
        self.last_system_prompt = ""

    @property
    def model_name(self) -> str:
        return "mock-llm"

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt or ""
        return "Câu trả lời thử nghiệm từ Mock LLM dựa trên tài liệu."

    def stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt or ""
        for token in ["Câu ", "trả ", "lời ", "thử ", "nghiệm."]:
            yield token


@pytest.fixture
def rag_pipeline(tmp_path):
    v_store = VectorStoreManager(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="test_collection"
    )
    emb = MockEmbeddingProvider()
    llm = MockLLMProvider()
    retriever = HybridRetriever(
        vector_store=v_store,
        embedding_provider=emb,
        coarse_k=4,
        top_k=2
    )
    chunker = StructureAwareChunker(parent_size=300, child_size=100)
    return DocumentRAGPipeline(
        embedding_provider=emb,
        llm_provider=llm,
        vector_store=v_store,
        retriever=retriever,
        chunker=chunker
    )


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


import sys
from pathlib import Path
from typing import List, Generator, Optional
import pytest

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings.embedding_provider import BaseEmbeddingProvider
from src.generation.llm_provider import BaseLLMProvider
from src.retrieval.vector_store import VectorStoreManager
from src.retrieval.retriever import HybridRetriever
from src.chunking.chunker import StructureAwareChunker
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

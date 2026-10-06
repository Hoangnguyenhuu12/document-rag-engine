"""
Unit tests for OpenAI Embedding and LLM Providers using Mock httpx clients.
"""

from unittest.mock import patch, MagicMock
import pytest

from src.embeddings.embedding_provider import OpenAIEmbeddingProvider, EmbeddingFactory
from src.generation.llm_provider import OpenAILLMProvider, LLMFactory


def test_openai_embedding_provider_query():
    provider = OpenAIEmbeddingProvider(
        api_key="sk-test-key",
        model="text-embedding-3-small",
        dimension=1536
    )

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "data": [{"embedding": [0.1] * 1536, "index": 0}]
    }

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        emb = provider.embed_query("Xin chào thế giới")
        assert len(emb) == 1536
        assert emb[0] == 0.1
        mock_post.assert_called_once()


def test_openai_embedding_provider_documents():
    provider = OpenAIEmbeddingProvider(api_key="sk-test-key")

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "data": [
            {"embedding": [0.2] * 1536, "index": 1},
            {"embedding": [0.1] * 1536, "index": 0}
        ]
    }

    with patch("httpx.Client.post", return_value=mock_resp):
        embs = provider.embed_documents(["doc 0", "doc 1"])
        assert len(embs) == 2
        # Verify ordering was preserved by index
        assert embs[0][0] == 0.1
        assert embs[1][0] == 0.2


def test_openai_llm_provider_generate():
    provider = OpenAILLMProvider(api_key="sk-test-key", model="gpt-4o-mini")

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "choices": [
            {"message": {"content": "Phản hồi thử nghiệm từ OpenAI."}}
        ]
    }

    with patch("httpx.Client.post", return_value=mock_resp):
        ans = provider.generate("Thời tiết hôm nay?", system_prompt="Bạn là trợ lý.")
        assert ans == "Phản hồi thử nghiệm từ OpenAI."


def test_factories_support_openai():
    emb = EmbeddingFactory.create("openai", api_key="test-key")
    assert isinstance(emb, OpenAIEmbeddingProvider)

    llm = LLMFactory.create("openai", api_key="test-key")
    assert isinstance(llm, OpenAILLMProvider)

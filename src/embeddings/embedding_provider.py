"""
Embedding Provider Abstraction for Document RAG Engine.
Supports Ollama (offline local), Gemini Cloud, and OpenAI.
"""

from abc import ABC, abstractmethod
import os
import time
from typing import List, Dict, Any, Optional
import httpx


class BaseEmbeddingProvider(ABC):
    """Abstract interface for dense vector embedding models."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """Local embedding via Ollama API (/api/embeddings)."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        dimension: int = 768
    ):
        self._model = model
        self.base_url = base_url.rstrip("/")
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model

    def embed_query(self, text: str) -> List[float]:
        url = f"{self.base_url}/api/embeddings"
        payload = {"model": self._model, "prompt": text}
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        url = f"{self.base_url}/api/embeddings"
        with httpx.Client(timeout=60.0) as client:
            for text in texts:
                payload = {"model": self._model, "prompt": text}
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                embeddings.append(resp.json()["embedding"])
        return embeddings


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """Google Gemini embedding provider with auto-retry."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "models/gemini-embedding-2",
        dimension: int = 768
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self._model = model
        self._dim = dimension
        if self.api_key:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model

    def embed_query(self, text: str) -> List[float]:
        import google.generativeai as genai
        res = genai.embed_content(
            model=self._model,
            content=text,
            task_type="retrieval_query",
            output_dimensionality=self._dim
        )
        return res["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        import google.generativeai as genai
        res = genai.embed_content(
            model=self._model,
            content=texts,
            task_type="retrieval_document",
            output_dimensionality=self._dim
        )
        return res["embedding"]


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider via REST API using httpx."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        dimension: int = 1536,
        base_url: str = "https://api.openai.com/v1"
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._model = model
        self._dim = dimension
        self.base_url = base_url.rstrip("/")

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def embed_query(self, text: str) -> List[float]:
        url = f"{self.base_url}/embeddings"
        payload: Dict[str, Any] = {
            "model": self._model,
            "input": text,
        }
        if "text-embedding-3" in self._model:
            payload["dimensions"] = self._dim

        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, headers=self._get_headers(), json=payload)
            resp.raise_for_status()
            data = resp.json()["data"]
            return data[0]["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        url = f"{self.base_url}/embeddings"
        payload: Dict[str, Any] = {
            "model": self._model,
            "input": texts,
        }
        if "text-embedding-3" in self._model:
            payload["dimensions"] = self._dim

        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, headers=self._get_headers(), json=payload)
            resp.raise_for_status()
            data = resp.json()["data"]
            # Ensure proper ordering by index
            sorted_data = sorted(data, key=lambda x: x["index"])
            return [item["embedding"] for item in sorted_data]


class EmbeddingFactory:
    """Factory to instantiate configured embedding provider."""

    @staticmethod
    def create(provider_name: str = "ollama", **kwargs) -> BaseEmbeddingProvider:
        provider_name = provider_name.lower()
        if provider_name == "ollama":
            return OllamaEmbeddingProvider(
                model=kwargs.get("model", "nomic-embed-text"),
                base_url=kwargs.get("base_url", "http://localhost:11434"),
                dimension=kwargs.get("dimension", 768)
            )
        elif provider_name == "gemini":
            return GeminiEmbeddingProvider(
                api_key=kwargs.get("api_key"),
                model=kwargs.get("model", "models/gemini-embedding-2"),
                dimension=kwargs.get("dimension", 768)
            )
        elif provider_name == "openai":
            return OpenAIEmbeddingProvider(
                api_key=kwargs.get("api_key"),
                model=kwargs.get("model", "text-embedding-3-small"),
                dimension=kwargs.get("dimension", 1536),
                base_url=kwargs.get("base_url", "https://api.openai.com/v1")
            )
        else:
            raise ValueError(f"Unsupported embedding provider: '{provider_name}'")


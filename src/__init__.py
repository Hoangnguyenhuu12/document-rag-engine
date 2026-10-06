"""
Document RAG Engine - Independent Document Ingestion, Retrieval and Generation Platform.
"""

# Eagerly import lightweight schema contracts
from .schemas.rag_contracts import (
    IngestedDocument,
    IngestedPage,
    DocumentChunk,
    Citation,
    RAGQueryRequest,
    RAGQueryResponse,
)

__all__ = [
    "DocumentRAGPipeline",
    "RAGPipeline",
    "IngestedDocument",
    "IngestedPage",
    "DocumentChunk",
    "Citation",
    "RAGQueryRequest",
    "RAGQueryResponse",
    "RAGDocumentLoader",
    "StructureAwareChunker",
    "EmbeddingFactory",
    "LLMFactory",
]

_LAZY_IMPORTS = {
    "DocumentRAGPipeline": (".pipeline.rag_pipeline", "DocumentRAGPipeline"),
    "RAGPipeline": (".pipeline.rag_pipeline", "DocumentRAGPipeline"),
    "RAGDocumentLoader": (".ingestion.document_loader", "RAGDocumentLoader"),
    "StructureAwareChunker": (".chunking.chunker", "StructureAwareChunker"),
    "EmbeddingFactory": (".embeddings.embedding_provider", "EmbeddingFactory"),
    "LLMFactory": (".generation.llm_provider", "LLMFactory"),
}


def __getattr__(name: str):
    """PEP 562: Lazy load heavy modules to prevent tight coupling and import bottlenecks."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path, __package__)
        return getattr(module, attr_name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

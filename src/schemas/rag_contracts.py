"""
Standardized Data Schemas and Interfaces for Document RAG Engine.
Decoupled from OCR implementations and frontend frameworks.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class IngestedPage(BaseModel):
    """Normalized page or section representation accepted by RAG Engine."""
    page_number: int
    text: str
    location_label: Optional[str] = None
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestedDocument(BaseModel):
    """Standardized document envelope for ingestion into RAG Engine."""
    doc_id: str
    file_name: str
    total_pages: int
    pages: List[IngestedPage]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """Single vector indexable chunk with parent-child hierarchical metadata."""
    chunk_id: str
    doc_id: str
    file_name: str
    text: str
    parent_id: Optional[str] = None
    parent_text: Optional[str] = None
    chunk_type: str = Field(default="child", description="child or parent")
    page_number: int = 1
    location_label: Optional[str] = None
    struct_path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    """Precise source attribution bound to a document location."""
    doc_name: str
    page_number: int
    location_label: str
    struct_path: Optional[str] = None
    text_snippet: str
    score: float = 0.0
    distance: float = 0.0


class RAGQueryRequest(BaseModel):
    """Query payload for RAG retrieval and generation."""
    query: str
    top_k: int = 4
    mode: str = Field(default="fast", description="'fast' or 'thinking'")
    filter_doc_name: Optional[str] = None
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    evaluate: bool = Field(default=False, description="Run RAG Triad evaluation on answer and citations")


class RAGEvaluationResult(BaseModel):
    """Evaluation score breakdown for a single RAG transaction (RAG Triad)."""
    context_relevance: float = Field(ge=0.0, le=1.0, description="Relevance of citations to query")
    groundedness: float = Field(ge=0.0, le=1.0, description="Faithfulness of answer to citations (anti-hallucination)")
    answer_relevance: float = Field(ge=0.0, le=1.0, description="Relevance of answer to user query")
    overall_score: float = Field(ge=0.0, le=1.0, description="Harmonic average of the triad")
    details: Dict[str, Any] = Field(default_factory=dict)


class RAGQueryResponse(BaseModel):
    """Complete RAG response with answer, citations, and execution telemetry."""
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    mode: str = "fast"
    model_used: str = ""
    thinking_trace: Optional[str] = None
    latency_seconds: float = 0.0
    evaluation: Optional[RAGEvaluationResult] = None


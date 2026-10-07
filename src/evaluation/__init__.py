"""
Evaluation Package for Document RAG Engine.
Includes RAG Triad evaluators for Context Relevance, Groundedness, and Answer Relevance.
"""

from .rag_triad import RAGTriadEvaluator

__all__ = ["RAGTriadEvaluator"]

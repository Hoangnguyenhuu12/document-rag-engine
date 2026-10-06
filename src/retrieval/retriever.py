"""
Two-Stage Hybrid Retriever with BM25 Lexical Density, RRF Reranking,
and Parent-Child Context Expansion.
"""

import math
import re
from typing import List, Dict, Any, Optional

from ..schemas.rag_contracts import Citation
from ..embeddings.embedding_provider import BaseEmbeddingProvider
from .vector_store import VectorStoreManager


class HybridRetriever:
    """Combines vector semantic search with lexical density and RRF fusion."""

    def __init__(
        self,
        vector_store: VectorStoreManager,
        embedding_provider: BaseEmbeddingProvider,
        coarse_k: int = 12,
        top_k: int = 4,
        semantic_weight: float = 0.5,
        lexical_weight: float = 0.3,
        rrf_weight: float = 0.2
    ):
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.coarse_k = coarse_k
        self.top_k = top_k
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
        self.rrf_weight = rrf_weight

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [w.lower() for w in re.findall(r'\b\w+\b', text) if len(w) > 1]

    def _calculate_lexical_score(self, query: str, text: str, location_label: str = "") -> float:
        """Calculate word overlap and exact match density."""
        q_tokens = set(self._tokenize(query))
        if not q_tokens:
            return 0.0

        doc_tokens = self._tokenize(text)
        if not doc_tokens:
            return 0.0

        matches = sum(1 for t in q_tokens if t in doc_tokens)
        overlap_ratio = matches / len(q_tokens)

        # Bonus for presence in location_label or struct_path
        label_bonus = 0.2 if any(t in location_label.lower() for t in q_tokens) else 0.0
        return min(1.0, overlap_ratio + label_bonus)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_doc_name: Optional[str] = None
    ) -> List[Citation]:
        """Execute two-stage hybrid retrieval and return expanded parent Citations."""
        k = top_k or self.top_k
        coarse_limit = max(k * 3, self.coarse_k)

        # 1. Coarse vector retrieval
        candidates = self.vector_store.query_similar(
            query=query,
            embedding_provider=self.embedding_provider,
            top_k=coarse_limit,
            doc_name_filter=filter_doc_name
        )

        if not candidates:
            return []

        # 2. Score candidates with Hybrid Reranker
        scored_candidates = []
        for rank, cand in enumerate(candidates):
            v_score = cand["vector_score"]
            meta = cand["metadata"]
            loc_label = meta.get("location_label", "")
            l_score = self._calculate_lexical_score(query, cand["text"], loc_label)
            rrf_score = 1.0 / (60.0 + (rank + 1))

            combined_score = (
                self.semantic_weight * v_score +
                self.lexical_weight * l_score +
                self.rrf_weight * rrf_score
            )
            scored_candidates.append({**cand, "combined_score": combined_score})

        # Sort by combined rerank score descending
        scored_candidates.sort(key=lambda x: x["combined_score"], reverse=True)

        # 3. Context expansion & deduplication by parent_id
        seen_parents = set()
        citations: List[Citation] = []

        for cand in scored_candidates:
            meta = cand["metadata"]
            parent_id = meta.get("parent_id")
            parent_text = meta.get("parent_text") or cand["text"]

            # Avoid duplicating the same parent block
            dedup_key = parent_id if parent_id else cand["chunk_id"]
            if dedup_key in seen_parents:
                continue
            seen_parents.add(dedup_key)

            citations.append(
                Citation(
                    doc_name=meta.get("file_name", "document"),
                    page_number=meta.get("page_number", 1),
                    location_label=meta.get("location_label", "Trang 1"),
                    struct_path=meta.get("struct_path"),
                    text_snippet=parent_text,
                    score=round(cand["combined_score"], 4),
                    distance=round(cand["distance"], 4)
                )
            )

            if len(citations) >= k:
                break

        return citations

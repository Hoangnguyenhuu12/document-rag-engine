"""
Two-Stage Hybrid Retriever with Okapi BM25 Lexical Ranking, RRF Fusion,
and Parent-Child Context Expansion.
"""

import math
import re
from typing import List, Dict, Any, Optional

from ..schemas.rag_contracts import Citation
from ..embeddings.embedding_provider import BaseEmbeddingProvider
from .vector_store import VectorStoreManager


class OkapiBM25Scorer:
    """
    Standard Okapi BM25 Scorer for lexical re-ranking.
    Computes inverse document frequency (IDF), term saturation (k1),
    and document length normalization (b).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Unicode-aware tokenization supporting Vietnamese and English."""
        return [w.lower() for w in re.findall(r'\b\w+\b', text, re.UNICODE) if w.strip()]

    def score_corpus(
        self,
        query: str,
        corpus_texts: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> List[float]:
        """Compute normalized [0.0, 1.0] Okapi BM25 scores for candidate documents."""
        q_tokens = self.tokenize(query)
        N = len(corpus_texts)
        if not q_tokens or N == 0:
            return [0.0] * N

        doc_tokens_list = [self.tokenize(t) for t in corpus_texts]
        doc_lens = [len(tokens) for tokens in doc_tokens_list]
        avgdl = sum(doc_lens) / N if N > 0 else 1.0
        if avgdl == 0:
            avgdl = 1.0

        # Document frequencies
        doc_freqs = []
        df: Dict[str, int] = {}
        for tokens in doc_tokens_list:
            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            doc_freqs.append(tf)
            for t in tf.keys():
                df[t] = df.get(t, 0) + 1

        # Calculate Robertson-Spärck Jones IDF
        idf: Dict[str, float] = {}
        for qt in set(q_tokens):
            n_q = df.get(qt, 0)
            idf[qt] = math.log((N - n_q + 0.5) / (n_q + 0.5) + 1.0)

        # Raw BM25 scores
        raw_scores: List[float] = []
        for i, tf in enumerate(doc_freqs):
            doc_len = doc_lens[i]
            score = 0.0
            for qt in q_tokens:
                f_qt = tf.get(qt, 0)
                if f_qt > 0:
                    numerator = f_qt * (self.k1 + 1.0)
                    denominator = f_qt + self.k1 * (1.0 - self.b + self.b * (doc_len / avgdl))
                    score += idf[qt] * (numerator / denominator)

            # Structural bonus for exact matches in location_label or struct_path
            if metadata_list and i < len(metadata_list):
                meta = metadata_list[i]
                loc_text = f"{meta.get('location_label', '')} {meta.get('struct_path', '')}".lower()
                matches = sum(1 for qt in set(q_tokens) if qt in loc_text)
                if matches > 0:
                    score += 0.3 * matches

            raw_scores.append(score)

        # Min-max normalization into [0.0, 1.0]
        max_s = max(raw_scores) if raw_scores else 0.0
        min_s = min(raw_scores) if raw_scores else 0.0

        if max_s <= 0.0:
            return [0.0] * N

        if max_s == min_s:
            return [1.0 if s > 0 else 0.0 for s in raw_scores]

        return [round((s - min_s) / (max_s - min_s), 4) for s in raw_scores]


class HybridRetriever:
    """Combines vector semantic search with Okapi BM25 and RRF fusion."""

    def __init__(
        self,
        vector_store: VectorStoreManager,
        embedding_provider: BaseEmbeddingProvider,
        coarse_k: int = 12,
        top_k: int = 4,
        semantic_weight: float = 0.5,
        lexical_weight: float = 0.3,
        rrf_weight: float = 0.2,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75
    ):
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.coarse_k = coarse_k
        self.top_k = top_k
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
        self.rrf_weight = rrf_weight
        self.bm25_scorer = OkapiBM25Scorer(k1=bm25_k1, b=bm25_b)

    def _calculate_lexical_score(self, query: str, text: str, location_label: str = "") -> float:
        """Single-document lexical score backward compatibility helper."""
        scores = self.bm25_scorer.score_corpus(
            query=query,
            corpus_texts=[text],
            metadata_list=[{"location_label": location_label}]
        )
        return scores[0] if scores else 0.0

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

        # 2. Score candidates with Okapi BM25 Lexical Ranking
        corpus_texts = [cand["text"] for cand in candidates]
        meta_list = [cand["metadata"] for cand in candidates]
        bm25_scores = self.bm25_scorer.score_corpus(query, corpus_texts, meta_list)

        scored_candidates = []
        for rank, cand in enumerate(candidates):
            v_score = cand["vector_score"]
            l_score = bm25_scores[rank]
            rrf_score = 1.0 / (60.0 + (rank + 1))

            combined_score = (
                self.semantic_weight * v_score +
                self.lexical_weight * l_score +
                self.rrf_weight * rrf_score
            )
            scored_candidates.append({
                **cand,
                "combined_score": combined_score,
                "bm25_score": l_score
            })

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

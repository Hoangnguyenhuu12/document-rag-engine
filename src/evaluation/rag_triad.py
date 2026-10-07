"""
RAG Triad Evaluator for Document RAG Engine.
Evaluates Context Relevance, Groundedness (Anti-Hallucination), and Answer Relevance.
"""

import re
from typing import List, Dict, Any, Optional
from ..schemas.rag_contracts import Citation, RAGEvaluationResult


class RAGTriadEvaluator:
    """Evaluates quality and faithfulness of RAG responses."""

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [w.lower() for w in re.findall(r'\b\w+\b', text, re.UNICODE) if len(w) > 1]

    def evaluate_context_relevance(self, query: str, citations: List[Citation]) -> float:
        """Measure how relevant retrieved citations are to the query."""
        if not citations:
            return 0.0

        q_tokens = set(self._tokenize(query))
        if not q_tokens:
            return 0.5

        scores = []
        for c in citations:
            c_tokens = set(self._tokenize(c.text_snippet))
            if not c_tokens:
                scores.append(0.0)
                continue
            overlap = len(q_tokens.intersection(c_tokens)) / len(q_tokens)
            scores.append(min(1.0, overlap * 1.5))  # scaled for recall

        # Average relevance across top citations
        avg_score = sum(scores) / len(scores) if scores else 0.0
        return round(min(1.0, max(0.0, avg_score)), 4)

    def evaluate_groundedness(self, answer: str, citations: List[Citation]) -> float:
        """Measure faithfulness of answer to retrieved citations (detects hallucination)."""
        if not answer or not answer.strip():
            return 0.0

        # Special case: Explicit refusal / lack of information in documents adheres to principles
        lower_ans = answer.lower()
        if "không chứa thông tin" in lower_ans or "không tìm thấy" in lower_ans or "chưa cung cấp" in lower_ans:
            return 1.0

        if not citations:
            return 0.0

        # Combined text from citations
        citations_text = " ".join([c.text_snippet for c in citations])
        citation_tokens = set(self._tokenize(citations_text))
        if not citation_tokens:
            return 0.0

        # Split answer into sentences
        sentences = [s.strip() for s in re.split(r'(?<=[.!?\n])\s+', answer) if len(s.strip()) > 10]
        if not sentences:
            sentences = [answer.strip()]

        supported_sentences = 0
        for sent in sentences:
            s_tokens = set(self._tokenize(sent))
            if not s_tokens:
                supported_sentences += 1
                continue
            overlap = len(s_tokens.intersection(citation_tokens)) / len(s_tokens)
            # A sentence is considered grounded if >= 45% of its informative words come from the citations
            if overlap >= 0.40:
                supported_sentences += 1

        score = supported_sentences / len(sentences)
        return round(min(1.0, max(0.0, score)), 4)

    def evaluate_answer_relevance(self, query: str, answer: str) -> float:
        """Measure how well the answer addresses the question."""
        if not answer or not answer.strip():
            return 0.0

        q_tokens = set(self._tokenize(query))
        if not q_tokens:
            return 0.5

        a_tokens = set(self._tokenize(answer))
        overlap = len(q_tokens.intersection(a_tokens)) / len(q_tokens)

        # Length penalty if answer is suspiciously short (e.g. less than 15 chars)
        length_penalty = 1.0 if len(answer.strip()) >= 20 else 0.5

        score = min(1.0, overlap * 1.8) * length_penalty
        return round(min(1.0, max(0.0, score)), 4)

    def evaluate(
        self,
        query: str,
        answer: str,
        citations: List[Citation]
    ) -> RAGEvaluationResult:
        """Perform comprehensive RAG Triad evaluation."""
        c_rel = self.evaluate_context_relevance(query, citations)
        grd = self.evaluate_groundedness(answer, citations)
        a_rel = self.evaluate_answer_relevance(query, answer)

        # Weighted composite score
        overall = round(0.35 * c_rel + 0.40 * grd + 0.25 * a_rel, 4)

        return RAGEvaluationResult(
            context_relevance=c_rel,
            groundedness=grd,
            answer_relevance=a_rel,
            overall_score=overall,
            details={
                "citations_count": len(citations),
                "answer_length": len(answer),
                "is_grounded": grd >= 0.70
            }
        )

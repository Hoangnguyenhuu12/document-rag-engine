"""
Central RAG Pipeline Orchestrator for Document RAG Engine.
Coordinates document ingestion, chunking, embedding, hybrid retrieval, and generation.
"""

import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator, Union

import yaml

from ..schemas.rag_contracts import (
    IngestedDocument,
    DocumentChunk,
    Citation,
    RAGQueryRequest,
    RAGQueryResponse,
    RAGEvaluationResult,
)
from ..ingestion.document_loader import RAGDocumentLoader
from ..chunking.chunker import StructureAwareChunker
from ..embeddings.embedding_provider import BaseEmbeddingProvider, EmbeddingFactory
from ..retrieval.vector_store import VectorStoreManager
from ..retrieval.retriever import HybridRetriever
from ..generation.llm_provider import BaseLLMProvider, LLMFactory
from ..evaluation.rag_triad import RAGTriadEvaluator

STRICT_RAG_SYSTEM_PROMPT = """\
Bạn là trợ lý AI chuyên nghiệp về khai thác tri thức tài liệu (Document Intelligence Assistant).
NHIỆM VỤ: Trả lời câu hỏi của người dùng DỰA HOÀN TOÀN VÀO CÁC ĐOẠN TRÍCH TÀI LIỆU ĐƯỢC CUNG CẤP DƯỚI ĐÂY.

NGUYÊN TẮC BẮT BUỘC:
1. KHÔNG BỊA ĐẶT (ZERO HALLUCINATION): Tuyệt đối không suy diễn hoặc đưa ra thông tin không có trong tài liệu.
2. NẾU THIẾU DỮ LIỆU: Nếu các đoạn trích không chứa đủ thông tin để trả lời, hãy nói rõ: "Tài liệu hiện có không chứa thông tin về câu hỏi này."
3. NGÔN NGỮ ĐỒNG BỘ: Nếu tài liệu là tiếng Anh nhưng người dùng hỏi bằng tiếng Việt, hãy trả lời bằng tiếng Việt chính xác, mượt mà dựa trên tài liệu đó.
4. TRẢ LỜI TRỰC DIỆN: Đi thẳng vào nội dung câu hỏi, nêu rõ các số liệu, căn cứ, điều khoản có trong tài liệu.
"""

THINKING_RAG_SYSTEM_PROMPT = """\
Bạn là chuyên gia phân tích và suy luận tri thức tài liệu chuyên sâu (Deep Document Reasoning Assistant).
NHIỆM VỤ: Trả lời câu hỏi của người dùng DỰA HOÀN TOÀN VÀO CÁC ĐOẠN TRÍCH TÀI LIỆU ĐƯỢC CUNG CẤP DƯỚI ĐÂY.

NGUYÊN TẮC SUY LUẬN & TRẢ LỜI:
1. SUY LUẬN CHUỖI TƯ DUY (CHAIN-OF-THOUGHT): Phân tích từng khía cạnh, đối chiếu chéo các điều khoản, bảng biểu liên quan trong tài liệu trước khi rút ra kết luận.
2. KHÔNG BỊA ĐẶT (ZERO HALLUCINATION): Mọi luận điểm, số liệu phải có căn cứ rõ ràng từ các đoạn trích. Tuyệt đối không suy đoán ngoài phạm vi tài liệu.
3. NẾU THIẾU THÔNG TIN: Hãy chỉ rõ phần nào tài liệu có đề cập và phần nào tài liệu chưa cung cấp.
4. TRÍCH DẪN RÕ RÀNG: Ghi rõ căn cứ (Điều, Khoản, Mục, Bảng biểu) tương ứng với từng kết luận.
"""


class DocumentRAGPipeline:
    """Independent end-to-end RAG pipeline."""

    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider,
        llm_provider: BaseLLMProvider,
        vector_store: Optional[VectorStoreManager] = None,
        retriever: Optional[HybridRetriever] = None,
        chunker: Optional[StructureAwareChunker] = None,
    ):
        self.embedding_provider = embedding_provider
        self.llm_provider = llm_provider
        self.vector_store = vector_store or VectorStoreManager()
        self.retriever = retriever or HybridRetriever(
            vector_store=self.vector_store,
            embedding_provider=self.embedding_provider
        )
        self.chunker = chunker or StructureAwareChunker()
        self.loader = RAGDocumentLoader()
        self.evaluator = RAGTriadEvaluator()

    @classmethod
    def from_configs(
        cls,
        embedding_config_path: Union[str, Path] = "configs/embedding.yaml",
        retrieval_config_path: Union[str, Path] = "configs/retrieval.yaml",
        llm_config_path: Union[str, Path] = "configs/llm.yaml",
        vector_store_dir: Optional[Union[str, Path]] = None,
    ) -> "DocumentRAGPipeline":
        """Instantiate pipeline from YAML configs."""
        with open(embedding_config_path, "r", encoding="utf-8") as f:
            emb_cfg = yaml.safe_load(f)
        with open(retrieval_config_path, "r", encoding="utf-8") as f:
            ret_cfg = yaml.safe_load(f)
        with open(llm_config_path, "r", encoding="utf-8") as f:
            llm_cfg = yaml.safe_load(f)

        def_emb = emb_cfg.get("default_provider", "ollama")
        emb_params = emb_cfg.get("providers", {}).get(def_emb, {})
        emb_provider = EmbeddingFactory.create(def_emb, **emb_params)

        def_llm = llm_cfg.get("default_provider", "gemini")
        llm_params = llm_cfg.get("providers", {}).get(def_llm, {})
        llm_provider = LLMFactory.create(def_llm, **llm_params)

        v_store_cfg = ret_cfg.get("vector_store", {})
        persist_dir = str(vector_store_dir) if vector_store_dir else v_store_cfg.get("persist_directory", "data/chroma_db")
        v_store = VectorStoreManager(
            persist_directory=persist_dir,
            collection_name=v_store_cfg.get("collection_name", "rag_knowledge_base")
        )

        r_cfg = ret_cfg.get("retrieval", {})
        rerank_cfg = ret_cfg.get("reranker", {})
        retriever = HybridRetriever(
            vector_store=v_store,
            embedding_provider=emb_provider,
            top_k=r_cfg.get("top_k", 4),
            coarse_k=r_cfg.get("coarse_k", 12),
            semantic_weight=rerank_cfg.get("semantic_weight", 0.5),
            lexical_weight=rerank_cfg.get("lexical_bm25_weight", 0.3),
            rrf_weight=rerank_cfg.get("rrf_weight", 0.2),
            bm25_k1=rerank_cfg.get("bm25_k1", 1.5),
            bm25_b=rerank_cfg.get("bm25_b", 0.75)
        )

        c_cfg = ret_cfg.get("chunking", {})
        chunker = StructureAwareChunker(
            parent_size=c_cfg.get("parent_size", 1200),
            child_size=c_cfg.get("child_size", 400),
            child_overlap=c_cfg.get("child_overlap", 0)
        )

        return cls(
            embedding_provider=emb_provider,
            llm_provider=llm_provider,
            vector_store=v_store,
            retriever=retriever,
            chunker=chunker
        )

    def index_document(self, doc: IngestedDocument, batch_size: int = 50) -> int:
        """Process document into chunks and persist into vector store."""
        chunks = self.chunker.chunk_document(doc)
        if chunks:
            self.vector_store.add_chunks(
                chunks=chunks,
                embedding_provider=self.embedding_provider,
                batch_size=batch_size
            )
        return len(chunks)

    def index_file(self, file_path: Union[str, Path], batch_size: int = 50) -> Dict[str, Any]:
        """Convenience method to load and index a native file (PDF, DOCX, XLSX)."""
        doc = self.loader.load_file(file_path)
        n_chunks = self.index_document(doc, batch_size=batch_size)
        return {
            "status": "success",
            "doc_id": doc.doc_id,
            "file_name": doc.file_name,
            "total_pages": doc.total_pages,
            "chunks_indexed": n_chunks
        }

    def index_ocr_json(
        self,
        ocr_data_or_path: Union[str, Path, Dict[str, Any]],
        source_path: Optional[str] = None,
        batch_size: int = 50
    ) -> Dict[str, Any]:
        """Convenience method to load and index standardized OCR JSON output."""
        doc = self.loader.load_from_ocr_json(ocr_data_or_path, source_path=source_path)
        n_chunks = self.index_document(doc, batch_size=batch_size)
        return {
            "status": "success",
            "doc_id": doc.doc_id,
            "file_name": doc.file_name,
            "total_pages": doc.total_pages,
            "chunks_indexed": n_chunks
        }

    def list_documents(self) -> List[Dict[str, Any]]:
        """List distinct documents and their chunk counts indexed in the vector store."""
        return self.vector_store.list_indexed_documents(self.embedding_provider.dimension)

    def delete_document(self, file_name: str) -> bool:
        """Delete all chunks of a specific document by file_name."""
        return self.vector_store.delete_document(file_name, self.embedding_provider.dimension)

    def clear_all_documents(self) -> bool:
        """Delete the entire collection and clear all documents."""
        return self.vector_store.clear_all_documents(self.embedding_provider.dimension)

    def _build_prompt(self, query: str, citations: List[Citation], conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Construct prompt with context blocks and optional chat history."""
        context_blocks = []
        for c in citations:
            header = f"[Tài liệu: {c.doc_name} | Vị trí: {c.location_label}]"
            context_blocks.append(f"{header}\n{c.text_snippet}")

        context_str = "\n\n---\n\n".join(context_blocks) if context_blocks else "Không tìm thấy đoạn trích phù hợp."

        history_str = ""
        if conversation_history:
            hist_lines = []
            for msg in conversation_history:
                role = "Người dùng" if msg.get("role") in ("user", "human") else "Trợ lý"
                content = msg.get("content", "")
                hist_lines.append(f"{role}: {content}")
            if hist_lines:
                history_str = "LỊCH SỬ HỘI THOẠI TRƯỚC ĐÓ:\n" + "\n".join(hist_lines) + "\n\n"

        return (
            f"NGỮ CẢNH TÀI LIỆU:\n\n{context_str}\n\n"
            f"{history_str}"
            f"CÂU HỎI: {query}\n\n"
            f"CÂU TRẢ LỜI:"
        )

    def query(self, request: RAGQueryRequest) -> RAGQueryResponse:
        """Execute complete RAG retrieval and generation."""
        start_time = time.time()

        # 1. Retrieve relevant parent citations
        citations = self.retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            filter_doc_name=request.filter_doc_name
        )

        # 2. Format prompt
        prompt = self._build_prompt(
            query=request.query,
            citations=citations,
            conversation_history=request.conversation_history
        )

        # 3. Select system prompt according to mode
        system_prompt = (
            THINKING_RAG_SYSTEM_PROMPT if request.mode == "thinking"
            else STRICT_RAG_SYSTEM_PROMPT
        )

        # 4. Call LLM Provider
        answer = self.llm_provider.generate(
            prompt=prompt,
            system_prompt=system_prompt
        )

        latency = time.time() - start_time

        eval_result = None
        if request.evaluate:
            eval_result = self.evaluator.evaluate(
                query=request.query,
                answer=answer,
                citations=citations
            )

        return RAGQueryResponse(
            answer=answer,
            citations=citations,
            mode=request.mode,
            model_used=self.llm_provider.model_name,
            latency_seconds=round(latency, 2),
            evaluation=eval_result
        )

    def evaluate_response(
        self,
        query: str,
        response: RAGQueryResponse
    ) -> RAGEvaluationResult:
        """Evaluate an existing query response using RAG Triad."""
        eval_res = self.evaluator.evaluate(
            query=query,
            answer=response.answer,
            citations=response.citations
        )
        response.evaluation = eval_res
        return eval_res


    def stream_query(
        self,
        request: RAGQueryRequest
    ) -> Generator[Union[str, List[Citation]], None, None]:
        """Stream RAG response tokens and yield citations at the start."""
        citations = self.retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            filter_doc_name=request.filter_doc_name
        )

        # Yield citations first so UI can render source cards
        yield citations

        prompt = self._build_prompt(
            query=request.query,
            citations=citations,
            conversation_history=request.conversation_history
        )

        system_prompt = (
            THINKING_RAG_SYSTEM_PROMPT if request.mode == "thinking"
            else STRICT_RAG_SYSTEM_PROMPT
        )

        for token in self.llm_provider.stream(prompt=prompt, system_prompt=system_prompt):
            yield token


# Backward-compatible alias
RAGPipeline = DocumentRAGPipeline


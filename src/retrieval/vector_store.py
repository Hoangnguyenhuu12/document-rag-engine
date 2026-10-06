"""
ChromaDB Vector Store Manager for Document RAG Engine.
Handles dense vector indexing, persistence, and batch persistence.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import chromadb
from chromadb.config import Settings

from ..schemas.rag_contracts import DocumentChunk
from ..embeddings.embedding_provider import BaseEmbeddingProvider


class VectorStoreManager:
    """Manages vector embeddings and metadata in ChromaDB."""

    def __init__(
        self,
        persist_directory: str = "data/chroma_db",
        collection_name: str = "rag_knowledge_base"
    ):
        self.persist_directory = str(Path(persist_directory).resolve())
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection_name = collection_name
        self._collection = None

    def get_collection(self, embedding_dimension: int):
        """Get or create collection tagged with embedding dimension."""
        coll_name = f"{self.collection_name}_{embedding_dimension}"
        if self._collection is None or self._collection.name != coll_name:
            self._collection = self._client.get_or_create_collection(
                name=coll_name,
                metadata={"hnsw:space": "cosine", "dimension": embedding_dimension}
            )
        return self._collection

    def get_existing_chunk_ids(self, doc_id: str, embedding_dim: int) -> Set[str]:
        """Fetch IDs of chunks already indexed for this doc_id."""
        coll = self.get_collection(embedding_dim)
        try:
            results = coll.get(where={"doc_id": doc_id}, include=[])
            return set(results.get("ids", []))
        except Exception:
            return set()

    def list_indexed_documents(self, embedding_dim: int) -> List[Dict[str, Any]]:
        """Fetch distinct documents and their chunk counts in ChromaDB."""
        coll = self.get_collection(embedding_dim)
        try:
            results = coll.get(include=["metadatas"])
            metas = results.get("metadatas", [])
            doc_counts: Dict[str, int] = {}
            for m in metas:
                fname = m.get("file_name", "unknown")
                doc_counts[fname] = doc_counts.get(fname, 0) + 1
            return [{"file_name": k, "chunk_count": v} for k, v in doc_counts.items()]
        except Exception:
            return []

    def delete_document(self, file_name: str, embedding_dim: int) -> bool:
        """Delete all chunks for a specific document from ChromaDB."""
        coll = self.get_collection(embedding_dim)
        try:
            coll.delete(where={"file_name": file_name})
            return True
        except Exception:
            return False

    def clear_all_documents(self, embedding_dim: int) -> bool:
        """Clear the entire knowledge base collection."""
        coll_name = f"{self.collection_name}_{embedding_dim}"
        try:
            self._client.delete_collection(name=coll_name)
            self._collection = None
            return True
        except Exception:
            return False

    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        embedding_provider: BaseEmbeddingProvider,
        batch_size: int = 50
    ):
        """Embed and persist chunks in batches."""
        if not chunks:
            return

        coll = self.get_collection(embedding_provider.dimension)
        existing_ids = self.get_existing_chunk_ids(chunks[0].doc_id, embedding_provider.dimension)
        pending = [c for c in chunks if c.chunk_id not in existing_ids]

        if not pending:
            return

        for i in range(0, len(pending), batch_size):
            batch = pending[i:i + batch_size]
            texts = [c.text for c in batch]
            ids = [c.chunk_id for c in batch]
            embeddings = embedding_provider.embed_documents(texts)
            metadatas = [
                {
                    "doc_id": c.doc_id,
                    "file_name": c.file_name,
                    "page_number": c.page_number,
                    "location_label": c.location_label or f"Trang {c.page_number}",
                    "struct_path": c.struct_path or "",
                    "parent_id": c.parent_id or "",
                    "parent_text": (c.parent_text or c.text)[:4000],  # Limit size for metadata safety
                    "chunk_type": c.chunk_type
                }
                for c in batch
            ]
            coll.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )

    def query_similar(
        self,
        query: str,
        embedding_provider: BaseEmbeddingProvider,
        top_k: int = 12,
        doc_name_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve coarse Top-K candidates for a query."""
        coll = self.get_collection(embedding_provider.dimension)
        q_emb = embedding_provider.embed_query(query)

        where_filter = {"file_name": doc_name_filter} if doc_name_filter else None
        results = coll.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        candidates: List[Dict[str, Any]] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return candidates

        for i in range(len(results["ids"][0])):
            cid = results["ids"][0][i]
            doc_text = results["documents"][0][i]
            meta = results["metadatas"][0][i]
            dist = results["distances"][0][i] if results.get("distances") else 0.0

            candidates.append({
                "chunk_id": cid,
                "text": doc_text,
                "metadata": meta,
                "distance": dist,
                "vector_score": max(0.0, 1.0 - dist)
            })

        return candidates

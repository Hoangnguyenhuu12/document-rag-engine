"""
Structure-Aware Parent-Child Document Chunker for Document RAG Engine.
Separates vector search units (sharp child chunks) from LLM reading units (full parent chunks).
"""

import hashlib
import re
from typing import List, Optional

from ..schemas.rag_contracts import IngestedPage, IngestedDocument, DocumentChunk

ARTICLE_PATTERN = re.compile(
    r'^(?:#+\s*)?(?:(?:\*\*)?\s*(?:Điều|Khoản|Mục|Chương|Phần|Tiết)\s+(\d+|[IVXLCDM]+|[A-Z])(?:\s*[:\.]\s*|\s+))',
    re.IGNORECASE | re.MULTILINE
)


class StructureAwareChunker:
    """Chunks documents into parent-child pairs preserving structural hierarchies."""

    def __init__(
        self,
        parent_size: int = 1200,
        child_size: int = 400,
        child_overlap: int = 0
    ):
        self.parent_size = parent_size
        self.child_size = child_size
        self.child_overlap = child_overlap

    @staticmethod
    def _extract_struct_path(text: str, default_label: str = "") -> str:
        """Extract structural hierarchy path (e.g. 'Điều 5 › Khoản 2')."""
        matches = ARTICLE_PATTERN.findall(text)
        if matches:
            first_match = ARTICLE_PATTERN.search(text)
            if first_match:
                matched_str = first_match.group(0).strip().lstrip('#').strip().replace('*', '')
                return f"{default_label} › {matched_str}" if default_label else matched_str
        return default_label

    def chunk_document(self, doc: IngestedDocument) -> List[DocumentChunk]:
        """Process all pages of an IngestedDocument into DocumentChunk list."""
        all_chunks: List[DocumentChunk] = []
        for page in doc.pages:
            chunks = self.chunk_page(
                page_text=page.text,
                doc_id=doc.doc_id,
                file_name=doc.file_name,
                page_number=page.page_number,
                location_label=page.location_label or f"Trang {page.page_number}"
            )
            all_chunks.extend(chunks)
        return all_chunks

    def chunk_page(
        self,
        page_text: str,
        doc_id: str,
        file_name: str,
        page_number: int,
        location_label: str
    ) -> List[DocumentChunk]:
        """Split a single page into parent-child chunk objects."""
        if not page_text or not page_text.strip():
            return []

        # If page is a markdown table, treat the entire table as one intact parent block
        if "| --- |" in page_text and len(page_text) <= 2500:
            parent_id = hashlib.md5(f"{doc_id}_{page_number}_tbl".encode()).hexdigest()[:12]
            return [
                DocumentChunk(
                    chunk_id=f"{parent_id}_c0",
                    doc_id=doc_id,
                    file_name=file_name,
                    text=page_text.strip(),
                    parent_id=parent_id,
                    parent_text=page_text.strip(),
                    chunk_type="child",
                    page_number=page_number,
                    location_label=location_label,
                    struct_path=location_label
                )
            ]

        # 1. Segment text into Parent blocks (~parent_size) along paragraph / section boundaries
        paragraphs = re.split(r'\n\s*\n', page_text)
        parent_blocks: List[str] = []
        current_parent: List[str] = []
        current_len = 0

        for para in paragraphs:
            p_strip = para.strip()
            if not p_strip:
                continue

            # Check if paragraph starts a new article or major section
            is_new_section = bool(ARTICLE_PATTERN.search(p_strip))
            if is_new_section and current_parent and current_len >= 300:
                parent_blocks.append("\n\n".join(current_parent))
                current_parent = [p_strip]
                current_len = len(p_strip)
            elif current_len + len(p_strip) > self.parent_size and current_parent:
                parent_blocks.append("\n\n".join(current_parent))
                current_parent = [p_strip]
                current_len = len(p_strip)
            else:
                current_parent.append(p_strip)
                current_len += len(p_strip)

        if current_parent:
            parent_blocks.append("\n\n".join(current_parent))

        # 2. Divide each Parent block into Child chunks (~child_size, overlap=0)
        chunks: List[DocumentChunk] = []

        for p_idx, parent_text in enumerate(parent_blocks):
            parent_id = hashlib.md5(f"{doc_id}_{page_number}_p{p_idx}".encode()).hexdigest()[:12]
            struct_path = self._extract_struct_path(parent_text, location_label)

            # Split parent text into sentences or small phrases
            sentences = re.split(r'(?<=[.!?\n])\s+', parent_text)
            current_child: List[str] = []
            child_len = 0
            child_sub_idx = 0

            for sent in sentences:
                s_strip = sent.strip()
                if not s_strip:
                    continue

                if child_len + len(s_strip) > self.child_size and current_child:
                    child_text = " ".join(current_child).strip()
                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"{parent_id}_c{child_sub_idx}",
                            doc_id=doc_id,
                            file_name=file_name,
                            text=child_text,
                            parent_id=parent_id,
                            parent_text=parent_text,
                            chunk_type="child",
                            page_number=page_number,
                            location_label=location_label,
                            struct_path=struct_path
                        )
                    )
                    child_sub_idx += 1
                    current_child = [s_strip]
                    child_len = len(s_strip)
                else:
                    current_child.append(s_strip)
                    child_len += len(s_strip)

            if current_child:
                child_text = " ".join(current_child).strip()
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{parent_id}_c{child_sub_idx}",
                        doc_id=doc_id,
                        file_name=file_name,
                        text=child_text,
                        parent_id=parent_id,
                        parent_text=parent_text,
                        chunk_type="child",
                        page_number=page_number,
                        location_label=location_label,
                        struct_path=struct_path
                    )
                )

        return chunks

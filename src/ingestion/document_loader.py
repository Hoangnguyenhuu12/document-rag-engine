"""
Document Loader and Parser for Document RAG Engine.
Ingests multi-format documents (PDF digital, DOCX, XLSX) and standardized OCR JSON outputs.
Strictly decoupled from OCR models and Vision APIs.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import List, Union, Dict, Any, Optional

import pymupdf as fitz
from ..schemas.rag_contracts import IngestedPage, IngestedDocument


class RAGDocumentLoader:
    """Document Loader for RAG ingestion."""

    @staticmethod
    def _generate_doc_id(file_path: Path) -> str:
        unique_str = f"{file_path.name}_{file_path.stat().st_size}"
        return hashlib.md5(unique_str.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub(r"\r\n|\r", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def load_from_ocr_json(
        self,
        ocr_json_path_or_dict: Union[str, Path, Dict[str, Any]],
        source_path: Optional[Union[str, Path]] = None
    ) -> IngestedDocument:
        """
        Ingest a document directly from a standardized OCR Engine output JSON.
        Does not perform any OCR.
        """
        if isinstance(ocr_json_path_or_dict, (str, Path)):
            path = Path(ocr_json_path_or_dict)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            doc_id = hashlib.md5(path.name.encode("utf-8")).hexdigest()[:12]
            doc_name = data.get("document_name", path.name)
        else:
            data = ocr_json_path_or_dict
            doc_name = data.get("document_name") or (Path(source_path).name if source_path else "document.pdf")
            doc_id = hashlib.md5(doc_name.encode("utf-8")).hexdigest()[:12]
        pages_raw = data.get("pages", [])
        ingested_pages: List[IngestedPage] = []

        for p in pages_raw:
            p_num = p.get("page_number", 1)
            p_text = p.get("text", "")
            ingested_pages.append(
                IngestedPage(
                    page_number=p_num,
                    text=f"# Tài liệu: {doc_name} (Trang {p_num})\n\n{p_text}",
                    location_label=f"Trang {p_num}",
                    tables=p.get("tables", []),
                    metadata=p.get("metadata", {})
                )
            )

        return IngestedDocument(
            doc_id=doc_id,
            file_name=doc_name,
            total_pages=len(ingested_pages),
            pages=ingested_pages,
            metadata=data.get("metadata", {})
        )

    def load_digital_pdf(self, file_path: Union[str, Path], min_char_threshold: int = 15) -> IngestedDocument:
        """Extract native digital text and tables from PDF (no OCR)."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        doc_id = self._generate_doc_id(path)
        doc = fitz.open(path)
        total_pages = len(doc)
        pages: List[IngestedPage] = []

        for page_idx in range(total_pages):
            page = doc[page_idx]
            raw_text = self._clean_text(page.get_text("text"))
            if len(raw_text) >= min_char_threshold:
                pages.append(
                    IngestedPage(
                        page_number=page_idx + 1,
                        text=f"# Tài liệu: {path.name} (Trang {page_idx + 1})\n\n{raw_text}",
                        location_label=f"Trang {page_idx + 1}"
                    )
                )

        doc.close()
        return IngestedDocument(
            doc_id=doc_id,
            file_name=path.name,
            total_pages=total_pages,
            pages=pages
        )

    def load_docx(self, file_path: Union[str, Path], max_section_chars: int = 1500) -> IngestedDocument:
        """Extract structured sections and tables from Word (.docx)."""
        import docx
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        doc_id = self._generate_doc_id(path)
        doc = docx.Document(path)
        sections: List[IngestedPage] = []

        current_heading = "Mục mở đầu"
        current_content: List[str] = []
        section_idx = 1

        for elem in doc.element.body:
            tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            if tag == "p":
                para = docx.text.paragraph.Paragraph(elem, doc)
                txt = para.text.strip()
                if not txt:
                    continue
                style_name = para.style.name.lower() if para.style else ""
                if "heading" in style_name:
                    if current_content:
                        sec_text = "\n\n".join(current_content)
                        sections.append(
                            IngestedPage(
                                page_number=section_idx,
                                text=f"# Tài liệu: {path.name} ({current_heading})\n\n{sec_text}",
                                location_label=current_heading
                            )
                        )
                        section_idx += 1
                        current_content = []
                    current_heading = f"Mục: {txt}"
                else:
                    current_content.append(txt)
            elif tag == "tbl":
                tbl = docx.table.Table(elem, doc)
                table_lines = []
                for r_idx, row in enumerate(tbl.rows):
                    row_cells = [c.text.strip().replace("\n", " ").replace("|", "\\|") for c in row.cells]
                    table_lines.append("| " + " | ".join(row_cells) + " |")
                    if r_idx == 0:
                        table_lines.append("| " + " | ".join(["---"] * len(row_cells)) + " |")
                if table_lines:
                    current_content.append("\n" + "\n".join(table_lines) + "\n")

        if current_content:
            sec_text = "\n\n".join(current_content)
            sections.append(
                IngestedPage(
                    page_number=section_idx,
                    text=f"# Tài liệu: {path.name} ({current_heading})\n\n{sec_text}",
                    location_label=current_heading
                )
            )

        return IngestedDocument(
            doc_id=doc_id,
            file_name=path.name,
            total_pages=len(sections),
            pages=sections
        )

    def load_excel(self, file_path: Union[str, Path], rows_per_chunk: int = 15) -> IngestedDocument:
        """Extract Excel spreadsheet sheets as structured Markdown tables."""
        import openpyxl
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        doc_id = self._generate_doc_id(path)
        wb = openpyxl.load_workbook(path, data_only=True)
        sections: List[IngestedPage] = []
        sec_idx = 1

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            all_rows = list(ws.iter_rows(values_only=True))
            if not all_rows:
                continue

            header_row = [str(c).strip() if c is not None else f"Cột_{i+1}" for i, c in enumerate(all_rows[0])]
            data_rows = all_rows[1:]

            for r_start in range(0, len(data_rows), rows_per_chunk):
                batch = data_rows[r_start:r_start + rows_per_chunk]
                table_lines = [
                    "| " + " | ".join(header_row) + " |",
                    "| " + " | ".join(["---"] * len(header_row)) + " |"
                ]
                for r in batch:
                    r_cells = [str(c).strip().replace("\n", " ").replace("|", "\\|") if c is not None else "" for c in r]
                    while len(r_cells) < len(header_row):
                        r_cells.append("")
                    table_lines.append("| " + " | ".join(r_cells[:len(header_row)]) + " |")

                r_end = min(r_start + len(batch), len(data_rows))
                loc_label = f"Sheet '{sheet_name}' (Dòng {r_start + 1} - {r_end})"
                md_content = f"# Bảng dữ liệu: {path.name} - {loc_label}\n\n" + "\n".join(table_lines)

                sections.append(
                    IngestedPage(
                        page_number=sec_idx,
                        text=md_content,
                        location_label=loc_label
                    )
                )
                sec_idx += 1

        wb.close()
        return IngestedDocument(
            doc_id=doc_id,
            file_name=path.name,
            total_pages=len(sections),
            pages=sections
        )

    def load_file(self, file_path: Union[str, Path]) -> IngestedDocument:
        """Route file to appropriate loader based on file extension."""
        path = Path(file_path)
        ext = path.suffix.lower()
        if ext == ".pdf":
            return self.load_digital_pdf(path)
        elif ext in (".docx", ".doc"):
            return self.load_docx(path)
        elif ext in (".xlsx", ".xls"):
            return self.load_excel(path)
        else:
            raise ValueError(f"Unsupported file format for RAG ingestion: '{ext}'")


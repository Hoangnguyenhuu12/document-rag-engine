"""
Unit tests for RAG Ingestion layer.
"""

from src.ingestion.document_loader import RAGDocumentLoader


def test_load_from_ocr_json_dict():
    loader = RAGDocumentLoader()
    sample_ocr_data = {
        "document_name": "scanned_contract.pdf",
        "total_pages": 2,
        "pages": [
            {
                "page_number": 1,
                "text": "Nội dung hợp đồng trang 1...",
                "is_scanned": True,
                "tables": []
            },
            {
                "page_number": 2,
                "text": "Điều khoản thanh toán trang 2...",
                "is_scanned": True,
                "tables": []
            }
        ],
        "metadata": {"title": "Hợp đồng dịch vụ"}
    }

    doc = loader.load_from_ocr_json(sample_ocr_data)
    assert doc.file_name == "scanned_contract.pdf"
    assert doc.total_pages == 2
    assert len(doc.pages) == 2
    assert doc.pages[0].page_number == 1
    assert "Trang 1" in doc.pages[0].location_label
    assert "Hợp đồng dịch vụ" in doc.metadata.get("title", "")

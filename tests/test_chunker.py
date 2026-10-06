"""
Unit tests for StructureAwareChunker in Document RAG Engine.
"""

from src.chunking.chunker import StructureAwareChunker
from src.schemas.rag_contracts import IngestedPage, IngestedDocument


def test_chunker_parent_child_linking():
    chunker = StructureAwareChunker(parent_size=500, child_size=150)
    page_text = (
        "# Điều 1. Phạm vi điều chỉnh\n\n"
        "Quy chế này quy định về chế độ làm việc từ xa đối với toàn thể cán bộ nhân viên công ty. "
        "Mỗi nhân viên được phép đăng ký tối đa 2 ngày làm việc từ xa mỗi tuần sau khi được quản lý trực tiếp phê duyệt.\n\n"
        "# Điều 2. Trách nhiệm của nhân viên\n\n"
        "Nhân viên làm việc từ xa có trách nhiệm bảo đảm tiến độ công việc, duy trì kết nối mạng thông suốt "
        "và tham gia đầy đủ các cuộc họp trực tuyến định kỳ."
    )
    chunks = chunker.chunk_page(
        page_text=page_text,
        doc_id="doc1",
        file_name="quy_che.docx",
        page_number=1,
        location_label="Mục 1"
    )

    assert len(chunks) > 0
    for c in chunks:
        assert c.chunk_type == "child"
        assert c.parent_id is not None
        assert c.parent_text is not None
        assert len(c.text) <= len(c.parent_text)
        assert c.doc_id == "doc1"


def test_table_preservation_as_intact_parent():
    chunker = StructureAwareChunker()
    table_text = (
        "# Bảng lương\n\n"
        "| Chức danh | Lương cơ bản | Phụ cấp |\n"
        "| --- | --- | --- |\n"
        "| Kỹ sư AI | 35.000.000 | 5.000.000 |\n"
        "| Trưởng nhóm | 50.000.000 | 8.000.000 |"
    )
    chunks = chunker.chunk_page(
        page_text=table_text,
        doc_id="doc2",
        file_name="bang_luong.xlsx",
        page_number=1,
        location_label="Sheet 1"
    )

    assert len(chunks) == 1
    assert "| --- |" in chunks[0].text
    assert chunks[0].parent_text == chunks[0].text

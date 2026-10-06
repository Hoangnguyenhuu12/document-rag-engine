# Document RAG Engine

Hệ thống truy xuất thông tin và sinh phản hồi tri thức tài liệu (RAG) độc lập, tách rời hoàn toàn khỏi mô hình OCR theo đúng kiến trúc mô-đun tại `kehoach.md` (Mục 10).

---

## 1. Tính Năng Cốt Lõi

- **Độc Lập Hoàn Toàn Khỏi OCR**: Không chứa logic nhận diện hay phụ thuộc thư viện thị giác máy tính nặng. Nhận dữ liệu đầu vào qua schema chuẩn `IngestedDocument` hoặc đọc trực tiếp từ tệp JSON chuẩn do `document-ocr-engine` xuất ra.
- **Nạp Đa Định Dạng**: Hỗ trợ bóc tách tài liệu số PDF (PyMuPDF), Word (.docx) và Excel (.xlsx) bảo toàn cấu trúc bảng biểu Markdown.
- **Phân Đoạn Cấu Trúc Cha - Con (Structure-Aware Parent-Child Chunking)**:
  - Chunk con (~400 ký tự, `overlap = 0`): tạo vector sắc nét, tối ưu độ chính xác tra cứu.
  - Chunk cha (~1200 ký tự): giữ trọn vẹn ngữ cảnh điều khoản, mục lục, bảng biểu giúp LLM đọc hiểu trọn vẹn.
- **Truy Xuất Lai Hai Giai Đoạn (Two-Stage Hybrid Retrieval)**:
  - Giai đoạn 1: Vector Semantic Search quét rộng (Coarse search).
  - Giai đoạn 2: Tái xếp hạng (Hybrid Reranker) kết hợp Vector score + BM25 Lexical density + Reciprocal Rank Fusion (RRF).
- **Trừu Tượng Hóa Nhà Cung Cấp (Provider Abstraction)**:
  - `EmbeddingFactory`: Hỗ trợ Ollama Local (`nomic-embed-text`), Gemini Cloud (`gemini-embedding-2`), OpenAI (`text-embedding-3-small`).
  - `LLMFactory`: Hỗ trợ Ollama Local (`deepseek-r1:8b`, `qwen2.5:14b`), Gemini Cloud (`gemini-2.5-flash`), OpenAI (`gpt-4o-mini`).
- **Hỗ Trợ Streaming & Hai Chế Độ Suy Luận**:
  - `Fast Mode`: Tối ưu độ trễ cho tra cứu định nghĩa, trích xuất dữ kiện tức thì (`STRICT_RAG_SYSTEM_PROMPT`).
  - `Thinking Mode`: Suy luận chuỗi tư duy chuyên sâu (Chain-of-Thought), đối chiếu chéo nhiều điều khoản (`THINKING_RAG_SYSTEM_PROMPT`).
- **REST API Sẵn Sàng (FastAPI)**: Đầy đủ endpoints tải tài liệu, nạp OCR, truy vấn và Server-Sent Events (SSE) Streaming.

---

## 2. Cài Đặt

```bash
cd document-rag-engine
pip install -r requirements.txt
```

---

## 3. Khởi Chạy Web API (FastAPI)

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Các Endpoint Chính:
- `GET /health`: Kiểm tra trạng thái dịch vụ.
- `POST /rag/index`: Nạp tài liệu số (Upload file qua `multipart/form-data` hoặc truyền `file_path`).
- `POST /rag/index-ocr`: Nạp kết quả JSON chuẩn hóa từ OCR Engine.
- `POST /rag/query`: Truy vấn tri thức (Hỗ trợ lọc tài liệu, lịch sử hội thoại, mode `fast` / `thinking`).
- `POST /rag/stream`: Truy vấn trực tiếp trả về Server-Sent Events (SSE) token-by-token kèm danh sách Citations.

---

## 4. Sử Dụng Qua Dòng Lệnh (CLI)

```bash
# Nạp tài liệu số thông thường
python cli.py index sample.pdf
python cli.py index bao_cao.docx
python cli.py index bang_luong.xlsx

# Nạp tài liệu từ kết quả của OCR Engine
python cli.py index ocr_result.json --from-ocr

# Tra cứu tri thức (Fast Mode)
python cli.py query "Quy chế làm việc từ xa được quy định như thế nào?"

# Tra cứu với chế độ suy luận chuyên sâu (Thinking Mode)
python cli.py query "So sánh doanh thu quý 1 và quý 2" --top-k 6 --mode thinking
```

---

## 5. Sử Dụng Trong Python Code

```python
from src.pipeline.rag_pipeline import DocumentRAGPipeline
from src.schemas.rag_contracts import RAGQueryRequest

pipeline = DocumentRAGPipeline.from_configs()

# 1. Nạp tài liệu trực tiếp
pipeline.index_file("chinh_sach.pdf")

# 2. Hoặc nạp từ kết quả OCR JSON
pipeline.index_ocr_json("ocr_result.json")

# 3. Truy vấn kèm lịch sử hội thoại & Thinking Mode
request = RAGQueryRequest(
    query="Quy chế làm việc từ xa được quy định như thế nào?",
    mode="thinking",
    top_k=4,
    conversation_history=[
        {"role": "user", "content": "Công ty có chính sách làm việc linh hoạt không?"},
        {"role": "assistant", "content": "Có, công ty áp dụng hình thức hybrid."}
    ]
)
response = pipeline.query(request)

print("Câu trả lời:", response.answer)
for c in response.citations:
    print(f"Trích dẫn: {c.doc_name} - {c.location_label} (Score: {c.score})")
```

---

## 6. Chạy Kiểm Thử (Unit Tests)

```bash
pytest
```
Bộ kiểm thử tự động kiểm tra toàn diện: bóc tách tài liệu số, thuật toán chunking Cha - Con, nhà cung cấp OpenAI (mocked), truy xuất lai 2 giai đoạn, chế độ suy luận Thinking, và toàn bộ endpoint FastAPI.

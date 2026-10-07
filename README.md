# Document RAG Engine

Hệ thống truy xuất thông tin và sinh phản hồi tri thức tài liệu (RAG) độc lập, tách rời hoàn toàn khỏi mô hình OCR theo đúng kiến trúc mô-đun chuẩn.

---

## 1. Tính Năng Cốt Lõi

- **Độc Lập Hoàn Toàn Khỏi OCR**: Không chứa logic nhận diện hay phụ thuộc thư viện thị giác máy tính nặng. Nhận dữ liệu đầu vào qua schema chuẩn `IngestedDocument` hoặc đọc trực tiếp từ tệp JSON chuẩn do `document-ocr-engine` xuất ra.
- **Nạp Đa Định Dạng**: Hỗ trợ bóc tách tài liệu số PDF (PyMuPDF), Word (.docx) và Excel (.xlsx) bảo toàn cấu trúc bảng biểu Markdown.
- **Phân Đoạn Cấu Trúc Cha - Con (Structure-Aware Parent-Child Chunking)**:
  - Chunk con (~400 ký tự, `overlap = 0`): tạo vector sắc nét, tối ưu độ chính xác tra cứu.
  - Chunk cha (~1200 ký tự): giữ trọn vẹn ngữ cảnh điều khoản, mục lục, bảng biểu giúp LLM đọc hiểu trọn vẹn.
- **Truy Xuất Lai Hai Giai Đoạn (Two-Stage Hybrid Retrieval & Okapi BM25)**:
  - Giai đoạn 1: Vector Semantic Search quét rộng (Coarse search).
  - Giai đoạn 2: Tái xếp hạng (Hybrid Reranker) kết hợp Vector score + **Thuật toán Okapi BM25** ($k_1=1.5, b=0.75$) + Reciprocal Rank Fusion (RRF).
- **Bộ Đánh Giá Chất Lượng Phản Hồi (RAG Triad Evaluation)**:
  - Tích hợp tự động đánh giá: **Context Relevance**, **Groundedness (Chống ảo giác)**, và **Answer Relevance**.
- **Quản Lý Kho Tri Thức Toàn Diện (CRUD Document Management)**:
  - Hỗ trợ liệt kê tài liệu (`list`), xóa từng tài liệu (`delete`), hoặc dọn sạch toàn bộ kho (`clear`) qua cả REST API và CLI.
- **Trừu Tượng Hóa Nhà Cung Cấp (Provider Abstraction)**:
  - `EmbeddingFactory`: Hỗ trợ Ollama Local (`nomic-embed-text`), Gemini Cloud (`gemini-embedding-2`), OpenAI (`text-embedding-3-small`).
  - `LLMFactory`: Hỗ trợ Ollama Local (`deepseek-r1:8b`, `qwen2.5:14b`), Gemini Cloud (`gemini-2.5-flash`), OpenAI (`gpt-4o-mini`).
- **Hỗ Trợ Streaming & Hai Chế Độ Suy Luận**:
  - `Fast Mode`: Tối ưu độ trễ cho tra cứu định nghĩa, trích xuất dữ kiện tức thì (`STRICT_RAG_SYSTEM_PROMPT`).
  - `Thinking Mode`: Suy luận chuỗi tư duy chuyên sâu (Chain-of-Thought), đối chiếu chéo nhiều điều khoản (`THINKING_RAG_SYSTEM_PROMPT`).
- **REST API & Giao Diện Web Studio (FastAPI + Modern Web UI)**:
  - Sẵn sàng endpoints REST API tải tài liệu, nạp OCR, truy vấn và Server-Sent Events (SSE) Streaming.
  - Tích hợp sẵn Web Studio UI đẹp mắt tại `http://localhost:8000/` cho phép kéo thả upload file, chat tương tác và xem thẻ trích dẫn trực tiếp.

---

## 2. Cài Đặt

```bash
cd document-rag-engine
pip install -r requirements.txt
```

---

## 3. Khởi Chạy Web Studio & REST API (FastAPI)

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Truy cập trình duyệt tại: **`http://localhost:8000/`** hoặc **`http://localhost:8000/ui`** để trải nghiệm giao diện Web Studio trực quan.

### Các Endpoint Chính:
- `GET /` hoặc `GET /ui`: Giao diện Web Studio tương tác trực tiếp.
- `GET /health`: Kiểm tra trạng thái dịch vụ.
- `GET /rag/documents`: Liệt kê danh sách tài liệu và số chunks trong kho.
- `DELETE /rag/documents/{file_name}`: Xóa tài liệu cụ thể khỏi kho tri thức.
- `DELETE /rag/documents`: Dọn sạch toàn bộ kho tri thức.
- `POST /rag/index`: Nạp tài liệu số (Upload file qua `multipart/form-data` hoặc truyền `file_path`).
- `POST /rag/index-ocr`: Nạp kết quả JSON chuẩn hóa từ OCR Engine.
- `POST /rag/query`: Truy vấn tri thức (Hỗ trợ lọc tài liệu, lịch sử hội thoại, mode `fast` / `thinking`, đánh giá RAG Triad).
- `POST /rag/stream`: Truy vấn trực tiếp trả về Server-Sent Events (SSE) token-by-token kèm danh sách Citations.

---

## 4. Sử Dụng Qua Dòng Lệnh (CLI)

```bash
# 1. Nạp tài liệu số thông thường
python cli.py index sample.pdf
python cli.py index bao_cao.docx
python cli.py index bang_luong.xlsx

# 2. Nạp tài liệu từ kết quả của OCR Engine
python cli.py index ocr_result.json --from-ocr

# 3. Quản lý kho tri thức
python cli.py list
python cli.py delete sample.pdf
python cli.py clear --force

# 4. Tra cứu tri thức (Fast Mode)
python cli.py query "Quy chế làm việc từ xa được quy định như thế nào?"

# 5. Tra cứu với chế độ suy luận chuyên sâu & đánh giá chất lượng (RAG Triad)
python cli.py query "So sánh doanh thu quý 1 và quý 2" --top-k 6 --mode thinking --eval
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

# 3. Quản lý kho tài liệu
docs = pipeline.list_documents()
print("Các tài liệu trong kho:", docs)

# 4. Truy vấn kèm đánh giá RAG Triad
request = RAGQueryRequest(
    query="Quy chế làm việc từ xa được quy định như thế nào?",
    mode="thinking",
    top_k=4,
    evaluate=True
)
response = pipeline.query(request)

print("Câu trả lời:", response.answer)
if response.evaluation:
    print(f"Groundedness: {response.evaluation.groundedness * 100}%")
    print(f"Overall Score: {response.evaluation.overall_score * 100}%")
```

---

## 6. Chạy Kiểm Thử (Unit Tests)

```bash
pytest
```
Bộ kiểm thử tự động gồm 27 test cases bao phủ toàn diện: bóc tách tài liệu số, thuật toán chunking Cha - Con, nhà cung cấp OpenAI (mocked), truy xuất lai Okapi BM25 + Vector, bộ đánh giá RAG Triad, quản lý tài liệu CRUD, và toàn bộ endpoint FastAPI.

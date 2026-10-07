"""
Command-Line Interface (CLI) for Standalone Document RAG Engine.
Usage:
    python cli.py index sample.pdf
    python cli.py index ocr_result.json --from-ocr
    python cli.py query "Trí tuệ cảm xúc là gì?"
"""

import argparse
import json
import sys
from pathlib import Path

# Configure utf-8 encoding for stdout on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline.rag_pipeline import DocumentRAGPipeline
from src.schemas.rag_contracts import RAGQueryRequest


def get_pipeline() -> DocumentRAGPipeline:
    config_dir = Path(__file__).parent / "configs"
    return DocumentRAGPipeline.from_configs(
        embedding_config_path=config_dir / "embedding.yaml",
        retrieval_config_path=config_dir / "retrieval.yaml",
        llm_config_path=config_dir / "llm.yaml"
    )


def handle_index(args):
    pipeline = get_pipeline()
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Ingesting '{input_path.name}' into RAG Engine...")
    suffix = input_path.suffix.lower()

    if args.from_ocr or suffix == ".json":
        doc = pipeline.loader.load_from_ocr_json(input_path)
    elif suffix == ".pdf":
        doc = pipeline.loader.load_digital_pdf(input_path)
    elif suffix in (".docx", ".doc"):
        doc = pipeline.loader.load_docx(input_path)
    elif suffix in (".xlsx", ".xls"):
        doc = pipeline.loader.load_excel(input_path)
    else:
        print(f"Unsupported file format: {suffix}", file=sys.stderr)
        sys.exit(1)

    n_chunks = pipeline.index_document(doc)
    print(f"Successfully indexed '{doc.file_name}': {len(doc.pages)} pages -> {n_chunks} parent-child chunks.")


def handle_query(args):
    pipeline = get_pipeline()
    print(f"Querying: \"{args.question}\"...\n")

    request = RAGQueryRequest(
        query=args.question,
        top_k=args.top_k,
        mode=args.mode,
        evaluate=getattr(args, "evaluate", False)
    )
    response = pipeline.query(request)

    print("--- CÂU TRẢ LỜI ---")
    print(response.answer)
    print(f"\n[Model: {response.model_used} | Latency: {response.latency_seconds}s]")

    if response.citations:
        print("\n--- TRÍCH DẪN NGUỒN (CITATIONS) ---")
        for i, c in enumerate(response.citations):
            print(f"[{i+1}] {c.doc_name} - {c.location_label} (Score: {c.score})")
            print(f"    \"{c.text_snippet[:150]}...\"\n")

    if response.evaluation:
        ev = response.evaluation
        print("--- ĐÁNH GIÁ CHẤT LƯỢNG (RAG TRIAD) ---")
        print(f"• Context Relevance (Độ liên quan ngữ cảnh): {ev.context_relevance * 100:.1f}%")
        print(f"• Groundedness (Tính trung thực/chống ảo giác): {ev.groundedness * 100:.1f}%")
        print(f"• Answer Relevance (Độ phù hợp câu trả lời): {ev.answer_relevance * 100:.1f}%")
        print(f"• Overall Triad Score: {ev.overall_score * 100:.1f}%\n")



def handle_list(args):
    pipeline = get_pipeline()
    docs = pipeline.list_documents()
    print(f"\n--- DANH SÁCH TÀI LIỆU TRONG KHO (Tổng: {len(docs)}) ---")
    if not docs:
        print("Chưa có tài liệu nào được nạp vào kho tri thức.")
        return
    for i, d in enumerate(docs, 1):
        print(f"[{i}] {d['file_name']} (Số chunk: {d['chunk_count']})")
    print()


def handle_delete(args):
    pipeline = get_pipeline()
    target = args.file_name
    print(f"Đang xóa tài liệu '{target}' khỏi kho tri thức...")
    success = pipeline.delete_document(target)
    if success:
        print(f"Đã xóa thành công tài liệu '{target}'.")
    else:
        print(f"Không thể xóa hoặc không tìm thấy tài liệu '{target}'.", file=sys.stderr)
        sys.exit(1)


def handle_clear(args):
    if not args.force:
        confirm = input("CẢNH BÁO: Thao tác này sẽ xóa toàn bộ kho tri thức. Bạn có chắc không? (y/N): ")
        if confirm.lower() != "y":
            print("Đã hủy thao tác.")
            return

    pipeline = get_pipeline()
    print("Đang dọn sạch toàn bộ kho tri thức...")
    success = pipeline.clear_all_documents()
    if success:
        print("Đã dọn sạch toàn bộ kho tri thức thành công.")
    else:
        print("Lỗi khi dọn dẹp kho tri thức.", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Standalone Document RAG Engine CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Index command
    index_parser = subparsers.add_parser("index", help="Index a document or OCR JSON into Vector Store")
    index_parser.add_argument("input_file", type=str, help="Path to document (PDF, DOCX, XLSX, JSON)")
    index_parser.add_argument("--from-ocr", action="store_true", help="Treat JSON file as standardized OCR output")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query knowledge base")
    query_parser.add_argument("question", type=str, help="User query text")
    query_parser.add_argument("--top-k", type=int, default=4, help="Number of citations to retrieve")
    query_parser.add_argument("--mode", choices=["fast", "thinking"], default="fast", help="Inference mode")
    query_parser.add_argument("--eval", dest="evaluate", action="store_true", help="Run RAG Triad evaluation")


    # List command
    subparsers.add_parser("list", help="List all indexed documents in the knowledge base")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a specific document from the knowledge base")
    delete_parser.add_argument("file_name", type=str, help="File name of the document to delete")

    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear all documents from the knowledge base")
    clear_parser.add_argument("--force", "-f", action="store_true", help="Skip confirmation prompt")

    args = parser.parse_args()

    if args.command == "index":
        handle_index(args)
    elif args.command == "query":
        handle_query(args)
    elif args.command == "list":
        handle_list(args)
    elif args.command == "delete":
        handle_delete(args)
    elif args.command == "clear":
        handle_clear(args)



if __name__ == "__main__":
    main()

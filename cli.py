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
        mode=args.mode
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

    args = parser.parse_args()

    if args.command == "index":
        handle_index(args)
    elif args.command == "query":
        handle_query(args)


if __name__ == "__main__":
    main()

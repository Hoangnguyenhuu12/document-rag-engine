"""
FastAPI entrypoint for Document RAG Engine.
Provides HTTP REST endpoints for indexing, querying, and streaming.
"""

import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.pipeline.rag_pipeline import DocumentRAGPipeline, RAGPipeline
from src.schemas.rag_contracts import RAGQueryRequest

app = FastAPI(
    title="Document RAG Engine API",
    version="1.0.0",
    description="Standalone RAG engine service with hybrid retrieval, parent-child chunking, and multi-provider LLM support."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy or singleton pipeline instance
_pipeline: Optional[DocumentRAGPipeline] = None


def get_pipeline() -> DocumentRAGPipeline:
    global _pipeline
    if _pipeline is None:
        config_dir = BASE_DIR / "configs"
        _pipeline = DocumentRAGPipeline.from_configs(
            embedding_config_path=config_dir / "embedding.yaml",
            retrieval_config_path=config_dir / "retrieval.yaml",
            llm_config_path=config_dir / "llm.yaml"
        )
    return _pipeline


class QueryRequest(BaseModel):
    query: str
    top_k: int = 4
    mode: str = Field(default="fast", description="'fast' or 'thinking'")
    filter_doc_name: Optional[str] = None
    chat_history: Optional[List[Dict[str, str]]] = None


class IndexOCRRequest(BaseModel):
    ocr_data: Dict[str, Any]
    source_path: str = "ocr_document"


@app.get("/health")
def health():
    return {"status": "healthy", "service": "document-rag-engine"}


@app.post("/rag/index")
async def index_document(
    file: UploadFile = File(None),
    file_path: str = Form(None)
):
    """
    Index a native document file (PDF, DOCX, XLSX) into the RAG vector store.
    """
    pipe = get_pipeline()
    temp_path = None
    try:
        if file is not None:
            suffix = Path(file.filename or "doc.pdf").suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(file.file, tmp)
                temp_path = Path(tmp.name)
            target_path = temp_path
        elif file_path:
            target_path = Path(file_path)
            if not target_path.exists():
                raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        else:
            raise HTTPException(status_code=400, detail="Must provide either 'file' or 'file_path'")

        result = pipe.index_file(target_path)
        return result
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)


@app.post("/rag/index-ocr")
async def index_ocr(request: IndexOCRRequest):
    """
    Index standardized OCROutput JSON into RAG.
    """
    pipe = get_pipeline()
    result = pipe.index_ocr_json(request.ocr_data, source_path=request.source_path)
    return result


@app.post("/rag/query")
async def query_rag(request: QueryRequest):
    """
    Query the indexed knowledge base with hybrid retrieval and synthesis.
    """
    pipe = get_pipeline()
    rag_request = RAGQueryRequest(
        query=request.query,
        top_k=request.top_k,
        mode=request.mode,
        filter_doc_name=request.filter_doc_name,
        conversation_history=request.chat_history or []
    )
    result = pipe.query(rag_request)
    return result


@app.post("/rag/stream")
async def stream_rag(request: QueryRequest):
    """
    Stream RAG response tokens and citations via Server-Sent Events (SSE).
    """
    pipe = get_pipeline()
    rag_request = RAGQueryRequest(
        query=request.query,
        top_k=request.top_k,
        mode=request.mode,
        filter_doc_name=request.filter_doc_name,
        conversation_history=request.chat_history or []
    )

    def event_stream():
        generator = pipe.stream_query(rag_request)
        try:
            # First item yielded is the list of Citations
            citations = next(generator)
            citations_data = [c.model_dump() for c in citations]
            yield f"event: citations\ndata: {json.dumps(citations_data, ensure_ascii=False)}\n\n"

            # Subsequent items are generated token strings
            for token in generator:
                yield f"event: token\ndata: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"

            yield "event: done\ndata: [DONE]\n\n"
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'error': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

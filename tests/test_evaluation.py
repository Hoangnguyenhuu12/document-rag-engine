"""
Unit tests for RAG Triad Evaluation in Document RAG Engine.
Tests Context Relevance, Groundedness (Anti-Hallucination), and Answer Relevance.
"""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.schemas.rag_contracts import Citation, RAGQueryRequest, RAGQueryResponse, RAGEvaluationResult
from src.evaluation.rag_triad import RAGTriadEvaluator


def test_context_relevance():
    evaluator = RAGTriadEvaluator()
    query = "Quy định về thời gian làm việc từ xa"

    relevant_citations = [
        Citation(
            doc_name="policy.pdf",
            page_number=1,
            location_label="Điều 3",
            text_snippet="Thời gian làm việc từ xa tối đa 2 ngày một tuần theo quy định.",
            score=0.9
        )
    ]
    irrelevant_citations = [
        Citation(
            doc_name="finance.pdf",
            page_number=5,
            location_label="Mục 2",
            text_snippet="Ngân sách mua sắm trang thiết bị văn phòng năm 2024.",
            score=0.2
        )
    ]

    rel_score = evaluator.evaluate_context_relevance(query, relevant_citations)
    irrel_score = evaluator.evaluate_context_relevance(query, irrelevant_citations)

    assert rel_score > irrel_score
    assert rel_score > 0.5
    assert irrel_score < 0.3


def test_groundedness_supported_and_refusal():
    evaluator = RAGTriadEvaluator()
    citations = [
        Citation(
            doc_name="hr.docx",
            page_number=1,
            location_label="Điều 4",
            text_snippet="Nhân viên chính thức được hưởng 12 ngày nghỉ phép nguyên lương hàng năm.",
            score=0.95
        )
    ]

    # Faithful answer
    good_ans = "Nhân viên chính thức được hưởng 12 ngày nghỉ phép hàng năm nguyên lương."
    assert evaluator.evaluate_groundedness(good_ans, citations) >= 0.7

    # Hallucinated / Unsupported answer
    hallucinated_ans = "Nhân viên được đi du lịch châu Âu miễn phí và nhận tiền thưởng 100 triệu mỗi tháng."
    assert evaluator.evaluate_groundedness(hallucinated_ans, citations) < 0.3

    # Refusal adhering to zero hallucination
    refusal_ans = "Tài liệu hiện có không chứa thông tin về câu hỏi này."
    assert evaluator.evaluate_groundedness(refusal_ans, citations) == 1.0


def test_answer_relevance():
    evaluator = RAGTriadEvaluator()
    query = "Mức phụ cấp đi lại là bao nhiêu?"
    good_ans = "Mức phụ cấp đi lại cho cán bộ công tác là 500.000 đồng/ngày theo điều 7."
    bad_ans = "Thời tiết hôm nay tại Hà Nội rất đẹp và có nắng nhẹ."

    assert evaluator.evaluate_answer_relevance(query, good_ans) > evaluator.evaluate_answer_relevance(query, bad_ans)


def test_full_triad_evaluation_composite():
    evaluator = RAGTriadEvaluator()
    query = "Chính sách làm việc hybrid của công ty?"
    citations = [
        Citation(
            doc_name="handbook.pdf",
            page_number=2,
            location_label="Mục 2",
            text_snippet="Công ty áp dụng mô hình làm việc hybrid kết hợp làm việc tại văn phòng và từ xa.",
            score=0.9
        )
    ]
    answer = "Công ty áp dụng chính sách làm việc hybrid kết hợp giữa làm việc tại văn phòng và làm việc từ xa."

    result = evaluator.evaluate(query, answer, citations)
    assert isinstance(result, RAGEvaluationResult)
    assert result.context_relevance > 0.5
    assert result.groundedness >= 0.7
    assert result.answer_relevance > 0.5
    assert result.overall_score >= 0.6
    assert result.details["is_grounded"] is True


def test_pipeline_query_with_evaluate_flag(rag_pipeline):
    from src.schemas.rag_contracts import IngestedDocument, IngestedPage

    doc = IngestedDocument(
        doc_id="eval_doc_01",
        file_name="chinh_sach_eval.pdf",
        total_pages=1,
        pages=[IngestedPage(page_number=1, text="Nhân viên làm việc 40 giờ một tuần.", location_label="Trang 1")]
    )
    rag_pipeline.index_document(doc)

    req = RAGQueryRequest(
        query="Thời gian làm việc một tuần là bao nhiêu giờ?",
        evaluate=True
    )
    resp = rag_pipeline.query(req)

    assert resp.evaluation is not None
    assert isinstance(resp.evaluation, RAGEvaluationResult)
    assert resp.evaluation.overall_score >= 0.0


def test_api_query_with_evaluation():
    client = TestClient(app)
    mock_pipeline = MagicMock()
    mock_eval = RAGEvaluationResult(
        context_relevance=0.85,
        groundedness=0.95,
        answer_relevance=0.90,
        overall_score=0.90,
        details={"is_grounded": True}
    )
    mock_resp = RAGQueryResponse(
        answer="Thời gian làm việc 40h/tuần.",
        citations=[],
        mode="fast",
        model_used="mock-llm",
        latency_seconds=0.1,
        evaluation=mock_eval
    )
    mock_pipeline.query.return_value = mock_resp

    with patch("api.main.get_pipeline", return_value=mock_pipeline):
        payload = {
            "query": "Bao nhiêu giờ?",
            "top_k": 2,
            "evaluate": True
        }
        res = client.post("/rag/query", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "evaluation" in data
        assert data["evaluation"]["groundedness"] == 0.95
        assert data["evaluation"]["overall_score"] == 0.90

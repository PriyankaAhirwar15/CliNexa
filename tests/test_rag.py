"""
Unit Tests for FAISS Grounded RAG Healthcare System
"""

import pytest
from rag.retriever import ClinicalRAGSystem


def test_rag_retrieval_and_answer():
    rag = ClinicalRAGSystem()
    assert rag.is_indexed is True
    assert len(rag.chunks) > 0

    query = "What is the general physiological role of dietary fiber?"
    retrieved = rag.retrieve(query, top_k=2)
    assert len(retrieved) > 0
    assert "Fiber" in retrieved[0]["document_title"] or "Dietary" in retrieved[0]["document_title"] or retrieved[0]["score"] > 0

    res = rag.answer_query(query)
    assert res["success"] is True
    assert len(res["answer"]) > 50
    assert len(res["sources"]) > 0
    assert "safety_disclaimer" in res


def test_rag_empty_query():
    rag = ClinicalRAGSystem()
    res = rag.answer_query("")
    assert res["success"] is False

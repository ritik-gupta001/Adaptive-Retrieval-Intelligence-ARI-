"""
Module 0 tests.
Covers: settings validation (including the edge cases that doc-driven specs
tend to skip), and that GraphState's TypedDicts accept the shapes the rest
of the system will produce.
"""
import os
import sys

import pytest

sys.path.append(".")


# --- Settings -------------------------------------------------------------

def test_settings_load_with_defaults(monkeypatch):
    monkeypatch.delenv("BM25_WEIGHT", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    from app.config.settings import Settings

    s = Settings(_env_file=None)
    assert s.environment == "dev"
    assert s.vector_weight == 0.6


def test_settings_rejects_out_of_range_weight():
    from app.config.settings import Settings
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(_env_file=None, bm25_weight=1.5)


def test_settings_list_fields_parsing():
    from app.config.settings import Settings

    # Comma-separated string format
    s1 = Settings(_env_file=None, uploaded_file_extensions=".pdf,.txt,.docx", document_reference_keywords="doc,pdf,cv")
    assert s1.uploaded_file_extensions == [".pdf", ".txt", ".docx"]
    assert s1.document_reference_keywords == ["doc", "pdf", "cv"]

    # JSON array string format
    s2 = Settings(_env_file=None, uploaded_file_extensions='[".pdf", ".txt"]', document_reference_keywords='["doc"]')
    assert s2.uploaded_file_extensions == [".pdf", ".txt"]
    assert s2.document_reference_keywords == ["doc"]


def test_active_llm_key_raises_if_missing():
    from app.config.settings import Settings

    s = Settings(
        _env_file=None,
        llm_provider="anthropic",
        anthropic_api_key=None,
        openai_api_key=None,
        groq_api_key=None,
        google_api_key=None,
    )
    with pytest.raises(ValueError):
        s.active_llm_key()


def test_active_llm_key_resolves_correct_provider():
    from app.config.settings import Settings

    s = Settings(_env_file=None, llm_provider="openai", openai_api_key="sk-test")
    assert s.active_llm_key() == "sk-test"


def test_enabled_strategies_excludes_unconfigured_optional_strategies():
    from app.config.settings import Settings

    s = Settings(_env_file=None, tavily_api_key=None, graph_rag_enabled=False)
    assert "web_search" not in s.enabled_strategies
    assert "graph_rag" not in s.enabled_strategies
    assert "vector_search" in s.enabled_strategies


def test_enabled_strategies_includes_web_search_when_key_present():
    from app.config.settings import Settings

    s = Settings(_env_file=None, tavily_api_key="tvly-test")
    assert "web_search" in s.enabled_strategies


# --- GraphState -------------------------------------------------------------

def test_graph_state_accepts_minimal_shape():
    from app.graph.state import GraphState

    state: GraphState = {"question": "What is LangGraph?", "run_id": "abc123"}
    assert state["question"] == "What is LangGraph?"


def test_graph_state_accepts_full_pipeline_shape():
    from app.graph.state import GraphState

    state: GraphState = {
        "question": "Compare hybrid vs vector search",
        "original_question": "Compare hybrid vs vector search",
        "conversation_id": "conv-1",
        "run_id": "run-1",
        "attributes": {
            "intent": "comparison",
            "complexity": "medium",
            "freshness_needed": False,
            "requires_multiple_sources": True,
            "is_ambiguous": False,
        },
        "strategies": ["hybrid_search"],
        "retrieved_docs": [{"content": "...", "source": "doc1", "score": 0.9}],
        "confidence": {
            "confidence_score": 0.82,
            "hallucination_risk": 0.05,
            "retrieval_quality": 0.9,
            "reflection_score": 0.88,
            "citation_quality": 0.95,
            "num_sources": 3,
            "confidence_level": "high",
        },
        "retry_count": 0,
    }
    assert state["confidence"]["confidence_level"] == "high"
    assert state["attributes"]["intent"] == "comparison"


def test_graph_state_clarification_fields_present():
    """Edge case: ambiguous query should be representable even though the
    clarification node itself doesn't exist yet."""
    from app.graph.state import GraphState

    state: GraphState = {
        "question": "Tell me about it",
        "clarification_needed": True,
        "clarification_question": "Could you clarify what 'it' refers to?",
    }
    assert state["clarification_needed"] is True

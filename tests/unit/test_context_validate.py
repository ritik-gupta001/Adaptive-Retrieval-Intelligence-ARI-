"""
Unit tests for Module 5. LLM calls mocked.
"""
import sys

import pytest

sys.path.append(".")

from app.core.exceptions import LLMCallError
from app.nodes.context_validate import context_validate_node, _format_documents


def _good_response(**overrides):
    base = {
        "is_relevant": True,
        "relevance_score": 0.85,
        "coverage_score": 0.8,
        "missing_information": "",
        "has_duplicates": False,
        "issues": [],
        "recommendation": "proceed",
    }
    base.update(overrides)
    return base


# --- empty-docs short-circuit -----------------------------------------------

@pytest.mark.asyncio
async def test_empty_documents_skips_llm_and_recommends_rewrite(monkeypatch):
    async def fail_if_called(*a, **k):
        raise AssertionError("LLM should not be called with zero documents")

    monkeypatch.setattr("app.nodes.context_validate.call_llm_json", fail_if_called)

    state = {"question": "anything", "reranked_docs": []}
    result = await context_validate_node(state)

    assert result["validation"]["recommendation"] == "rewrite"
    assert result["validation"]["is_relevant"] is False


# --- normal validation path --------------------------------------------------

@pytest.mark.asyncio
async def test_relevant_documents_recommend_proceed(monkeypatch):
    async def fake_call_llm_json(prompt_name, **kwargs):
        assert prompt_name == "context_validator"
        return _good_response()

    monkeypatch.setattr("app.nodes.context_validate.call_llm_json", fake_call_llm_json)

    state = {
        "question": "What is LangGraph?",
        "reranked_docs": [{"content": "LangGraph is a library...", "source": "doc1", "score": 0.9}],
    }
    result = await context_validate_node(state)

    assert result["validation"]["recommendation"] == "proceed"
    assert result["validation"]["relevance_score"] == 0.85


@pytest.mark.asyncio
async def test_irrelevant_documents_recommend_rewrite(monkeypatch):
    async def fake_call_llm_json(prompt_name, **kwargs):
        return _good_response(
            is_relevant=False,
            relevance_score=0.1,
            recommendation="rewrite",
            issues=["documents discuss unrelated topic"],
        )

    monkeypatch.setattr("app.nodes.context_validate.call_llm_json", fake_call_llm_json)

    state = {
        "question": "What is LangGraph?",
        "reranked_docs": [{"content": "Unrelated content.", "source": "doc1", "score": 0.3}],
    }
    result = await context_validate_node(state)

    assert result["validation"]["recommendation"] == "rewrite"
    assert "validation:documents discuss unrelated topic" in result["issues_log"]


@pytest.mark.asyncio
async def test_recommends_change_strategy_for_shallow_coverage(monkeypatch):
    async def fake_call_llm_json(prompt_name, **kwargs):
        return _good_response(
            coverage_score=0.2,
            recommendation="change_strategy",
            missing_information="No discussion of multi-hop reasoning aspects.",
        )

    monkeypatch.setattr("app.nodes.context_validate.call_llm_json", fake_call_llm_json)

    state = {
        "question": "Research question",
        "reranked_docs": [{"content": "Partial info.", "source": "doc1", "score": 0.7}],
    }
    result = await context_validate_node(state)
    assert result["validation"]["recommendation"] == "change_strategy"


# --- LLM failure fallback ----------------------------------------------------

@pytest.mark.asyncio
async def test_llm_failure_degrades_to_proceed_not_crash(monkeypatch):
    async def failing_call_llm_json(*a, **k):
        raise LLMCallError("simulated provider outage")

    monkeypatch.setattr("app.nodes.context_validate.call_llm_json", failing_call_llm_json)

    state = {
        "question": "anything",
        "reranked_docs": [{"content": "some content", "source": "doc1", "score": 0.7}],
    }
    result = await context_validate_node(state)

    assert result["validation"]["recommendation"] == "proceed"
    assert any("unavailable" in i for i in result["validation"]["issues"])


@pytest.mark.asyncio
async def test_schema_violation_also_degrades_to_proceed(monkeypatch):
    """Edge case: LLM returns an invalid recommendation enum value."""
    async def fake_call_llm_json(*a, **k):
        return {
            "is_relevant": True,
            "relevance_score": 0.8,
            "coverage_score": 0.8,
            "recommendation": "not_a_real_option",
        }

    monkeypatch.setattr("app.nodes.context_validate.call_llm_json", fake_call_llm_json)

    state = {
        "question": "anything",
        "reranked_docs": [{"content": "some content", "source": "doc1", "score": 0.7}],
    }
    result = await context_validate_node(state)
    assert result["validation"]["recommendation"] == "proceed"


# --- _format_documents helper -------------------------------------------------

def test_format_documents_truncates_long_content():
    docs = [{"content": "x" * 1000, "source": "doc1"}]
    formatted = _format_documents(docs)
    assert len(formatted) < 700  # truncated well below the raw 1000 chars + wrapper text


def test_format_documents_includes_source_and_index():
    docs = [{"content": "hello", "source": "doc1"}, {"content": "world", "source": "doc2"}]
    formatted = _format_documents(docs)
    assert "[1]" in formatted and "[2]" in formatted
    assert "doc1" in formatted and "doc2" in formatted

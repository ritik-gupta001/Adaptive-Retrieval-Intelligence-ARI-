"""
Unit tests for the Query Rewrite node.
"""
import sys

import pytest

sys.path.append(".")

from app.core.exceptions import LLMCallError
from app.nodes.query_rewrite import query_rewrite_node, _STALE_FIELDS
from app.memory.store import ARIStore
import app.memory.store as store_mod


@pytest.fixture(autouse=True)
def reset_store(monkeypatch):
    monkeypatch.setattr(store_mod, "_store", ARIStore())


@pytest.mark.asyncio
async def test_increments_retry_count(monkeypatch):
    async def fake_call_llm(prompt, **kwargs):
        return "Rewritten question"

    monkeypatch.setattr("app.nodes.query_rewrite.call_llm", fake_call_llm)

    state = {"question": "original q", "retry_count": 0, "issues_log": []}
    result = await query_rewrite_node(state)
    assert result["retry_count"] == 1


@pytest.mark.asyncio
async def test_preserves_original_question_across_rewrites(monkeypatch):
    async def fake_call_llm(prompt, **kwargs):
        return "Better phrased question"

    monkeypatch.setattr("app.nodes.query_rewrite.call_llm", fake_call_llm)

    state = {"question": "original q", "original_question": "original q", "retry_count": 0, "issues_log": []}
    result = await query_rewrite_node(state)

    assert result["original_question"] == "original q"
    assert result["question"] == "Better phrased question"


@pytest.mark.asyncio
async def test_clears_all_stale_downstream_fields(monkeypatch):
    async def fake_call_llm(prompt, **kwargs):
        return "New question"

    monkeypatch.setattr("app.nodes.query_rewrite.call_llm", fake_call_llm)

    state = {
        "question": "q",
        "retry_count": 0,
        "issues_log": [],
        "retrieved_docs": [{"content": "stale doc"}],
        "answer": "stale answer",
        "confidence": {"confidence_score": 0.9},
    }
    result = await query_rewrite_node(state)

    for field in _STALE_FIELDS:
        assert result.get(field) is None, f"Expected {field} to be cleared"


@pytest.mark.asyncio
async def test_falls_back_to_original_question_on_llm_failure(monkeypatch):
    async def failing_call_llm(prompt, **kwargs):
        raise LLMCallError("provider down")

    monkeypatch.setattr("app.nodes.query_rewrite.call_llm", failing_call_llm)

    state = {
        "question": "current q",
        "original_question": "original q",
        "retry_count": 0,
        "issues_log": [],
    }
    result = await query_rewrite_node(state)

    assert result["question"] == "original q"  # fell back to original
    assert result["retry_count"] == 1          # retry count still incremented


@pytest.mark.asyncio
async def test_strips_quotes_from_rewritten_question(monkeypatch):
    async def fake_call_llm(prompt, **kwargs):
        return '"What is LangGraph persistence?"'  # LLM sometimes wraps in quotes

    monkeypatch.setattr("app.nodes.query_rewrite.call_llm", fake_call_llm)

    state = {"question": "q", "retry_count": 0, "issues_log": []}
    result = await query_rewrite_node(state)
    assert not result["question"].startswith('"')
    assert not result["question"].endswith('"')


@pytest.mark.asyncio
async def test_appends_rewrite_event_to_issues_log(monkeypatch):
    async def fake_call_llm(prompt, **kwargs):
        return "New question"

    monkeypatch.setattr("app.nodes.query_rewrite.call_llm", fake_call_llm)

    state = {"question": "q", "retry_count": 0, "issues_log": ["existing issue"]}
    result = await query_rewrite_node(state)

    assert "existing issue" in result["issues_log"]
    assert any("retry_1" in entry for entry in result["issues_log"])


@pytest.mark.asyncio
async def test_preserves_web_search_strategy_when_local_retrieval_failed(monkeypatch):
    """Critical: strategies must NOT be wiped when context_validate set web_search fallback.
    If cleared, the retry would re-enter router and pick local strategies again,
    defeating the entire local-failure recovery path (was Bug #1 in the audit).
    """
    async def fake_call_llm(prompt, **kwargs):
        return "New question"

    monkeypatch.setattr("app.nodes.query_rewrite.call_llm", fake_call_llm)

    state = {
        "question": "q",
        "retry_count": 0,
        "issues_log": [],
        "strategies": ["web_search"],
        "local_retrieval_failed": True,  # set by context_validate_node
    }
    result = await query_rewrite_node(state)
    # strategies must survive the rewrite when local retrieval failed
    assert result["strategies"] == ["web_search"], (
        "web_search fallback strategy was wiped by query_rewrite — this is the "
        "critical Bug #1 that breaks local-failure recovery"
    )


@pytest.mark.asyncio
async def test_clears_strategies_when_quality_improvement_retry(monkeypatch):
    """When local_retrieval_failed is False, strategies should be cleared so the
    router can re-decide with the rewritten query on the next pass.
    """
    async def fake_call_llm(prompt, **kwargs):
        return "New question"

    monkeypatch.setattr("app.nodes.query_rewrite.call_llm", fake_call_llm)

    state = {
        "question": "q",
        "retry_count": 0,
        "issues_log": [],
        "strategies": ["hybrid_search"],
        "local_retrieval_failed": False,  # quality retry, not fallback
    }
    result = await query_rewrite_node(state)
    # strategies should be cleared so the router re-decides
    assert result.get("strategies") is None

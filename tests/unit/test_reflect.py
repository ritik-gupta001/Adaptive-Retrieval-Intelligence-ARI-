"""
Unit tests for Module 7. LLM calls mocked.
"""
import sys

import pytest

sys.path.append(".")

from app.core.exceptions import LLMCallError
from app.nodes.reflect import reflect_node, _format_context


def _good_response(**overrides):
    base = {
        "is_supported": True,
        "hallucinations": [],
        "unsupported_claims": [],
        "missing_information": "",
        "incorrect_reasoning": [],
        "completeness_score": 0.9,
        "overall_score": 0.88,
        "should_retry": False,
        "reasoning": "Answer is well-supported.",
    }
    base.update(overrides)
    return base


# --- no-context short-circuit ------------------------------------------------

@pytest.mark.asyncio
async def test_no_context_skips_llm_and_signals_retry(monkeypatch):
    async def fail_if_called(*a, **k):
        raise AssertionError("LLM should not be called with no context")

    monkeypatch.setattr("app.nodes.reflect.call_llm_json", fail_if_called)

    state = {
        "question": "anything",
        "answer": "I don't have enough information.",
        "reranked_docs": [],
    }
    result = await reflect_node(state)

    assert result["reflection"]["should_retry"] is True
    assert result["reflection"]["completeness_score"] == 0.0
    assert result["reflection"]["is_supported"] is True  # vacuously


# --- normal reflection path -----------------------------------------------

@pytest.mark.asyncio
async def test_well_supported_answer_passes(monkeypatch):
    async def fake_call_llm_json(prompt_name, **kwargs):
        assert prompt_name == "reflection"
        return _good_response()

    monkeypatch.setattr("app.nodes.reflect.call_llm_json", fake_call_llm_json)

    state = {
        "question": "What is LangGraph?",
        "answer": "LangGraph is a library [Source 1].",
        "reranked_docs": [{"content": "LangGraph is a library for agents.", "source": "doc1"}],
    }
    result = await reflect_node(state)

    assert result["reflection"]["is_supported"] is True
    assert result["reflection"]["should_retry"] is False


@pytest.mark.asyncio
async def test_hallucination_detected_logged_in_issues(monkeypatch):
    async def fake_call_llm_json(prompt_name, **kwargs):
        return _good_response(
            is_supported=False,
            hallucinations=["claims LangGraph was released in 2019, not in context"],
            overall_score=0.3,
            should_retry=True,
        )

    monkeypatch.setattr("app.nodes.reflect.call_llm_json", fake_call_llm_json)

    state = {
        "question": "When was LangGraph released?",
        "answer": "LangGraph was released in 2019.",
        "reranked_docs": [{"content": "LangGraph is a library.", "source": "doc1"}],
    }
    result = await reflect_node(state)

    assert result["reflection"]["is_supported"] is False
    assert result["reflection"]["should_retry"] is True
    assert any("hallucinations" in entry for entry in result["issues_log"])


@pytest.mark.asyncio
async def test_unsupported_claims_logged_in_issues(monkeypatch):
    async def fake_call_llm_json(prompt_name, **kwargs):
        return _good_response(unsupported_claims=["mentions a feature not in context"])

    monkeypatch.setattr("app.nodes.reflect.call_llm_json", fake_call_llm_json)

    state = {
        "question": "q",
        "answer": "a",
        "reranked_docs": [{"content": "c", "source": "doc1"}],
    }
    result = await reflect_node(state)
    assert any("unsupported_claims" in entry for entry in result["issues_log"])


@pytest.mark.asyncio
async def test_low_completeness_can_signal_retry_even_if_supported(monkeypatch):
    """Edge case: answer is technically accurate but incomplete — should
    still be able to signal a retry."""
    async def fake_call_llm_json(prompt_name, **kwargs):
        return _good_response(
            is_supported=True,
            completeness_score=0.2,
            missing_information="Doesn't cover the comparison aspect of the question.",
            should_retry=True,
        )

    monkeypatch.setattr("app.nodes.reflect.call_llm_json", fake_call_llm_json)

    state = {
        "question": "Compare X and Y",
        "answer": "X is...",
        "reranked_docs": [{"content": "info about X only", "source": "doc1"}],
    }
    result = await reflect_node(state)
    assert result["reflection"]["should_retry"] is True
    assert result["reflection"]["is_supported"] is True


# --- LLM failure fallback ----------------------------------------------------

@pytest.mark.asyncio
async def test_llm_failure_degrades_to_cautious_defaults_not_crash(monkeypatch):
    async def failing_call_llm_json(*a, **k):
        raise LLMCallError("simulated provider outage")

    monkeypatch.setattr("app.nodes.reflect.call_llm_json", failing_call_llm_json)

    state = {
        "question": "q",
        "answer": "a",
        "reranked_docs": [{"content": "c", "source": "doc1"}],
    }
    result = await reflect_node(state)

    assert result["reflection"]["should_retry"] is False
    assert "unavailable" in result["reflection"]["reasoning"]


@pytest.mark.asyncio
async def test_schema_violation_also_degrades_gracefully(monkeypatch):
    """Edge case: LLM returns completeness_score out of the 0-1 range."""
    async def fake_call_llm_json(*a, **k):
        return {
            "is_supported": True,
            "completeness_score": 1.5,  # invalid, out of range
            "overall_score": 0.8,
            "should_retry": False,
        }

    monkeypatch.setattr("app.nodes.reflect.call_llm_json", fake_call_llm_json)

    state = {
        "question": "q",
        "answer": "a",
        "reranked_docs": [{"content": "c", "source": "doc1"}],
    }
    result = await reflect_node(state)
    assert result["reflection"]["reasoning"].startswith("Reflection agent unavailable")


# --- _format_context helper ---------------------------------------------------

def test_format_context_truncates_per_doc():
    docs = [{"content": "y" * 2000}]
    formatted = _format_context(docs)
    assert len(formatted) < 1100

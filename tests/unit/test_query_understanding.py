"""
Unit tests for the Query Understanding Agent.
LLM calls are mocked via monkeypatching app.nodes.query_understanding.call_llm_json
so these run fast, free, and deterministically in CI.
"""
import sys

import pytest

sys.path.append(".")

from app.core.exceptions import LLMCallError
from app.nodes.query_understanding import query_understanding_node


def _good_response(**overrides):
    base = {
        "intent": "factual",
        "complexity": "simple",
        "freshness_needed": False,
        "requires_multiple_sources": False,
        "requires_reasoning": False,
        "requires_comparison": False,
        "domain": "general knowledge",
        "entities": ["LangGraph"],
        "keywords": ["langgraph", "definition"],
        "reasoning": "Simple factual lookup.",
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_classifies_simple_factual_question(monkeypatch):
    async def fake_call_llm_json(prompt_name, **kwargs):
        assert prompt_name == "query_understanding"
        assert kwargs["question"] == "What is LangGraph?"
        return _good_response()

    monkeypatch.setattr(
        "app.nodes.query_understanding.call_llm_json", fake_call_llm_json
    )

    state = {"question": "What is LangGraph?"}
    result = await query_understanding_node(state)

    assert result["attributes"]["intent"] == "factual"
    assert result["attributes"]["complexity"] == "simple"
    assert result["attributes"]["is_ambiguous"] is False
    # node must never touch question/answer
    assert result["question"] == "What is LangGraph?"
    assert "answer" not in result


@pytest.mark.asyncio
async def test_never_generates_an_answer_field(monkeypatch):
    """Hard requirement from the spec: this node must not answer."""
    async def fake_call_llm_json(prompt_name, **kwargs):
        return _good_response()

    monkeypatch.setattr(
        "app.nodes.query_understanding.call_llm_json", fake_call_llm_json
    )

    state = {"question": "What is LangGraph?", "answer": ""}
    result = await query_understanding_node(state)
    assert result["answer"] == ""  # untouched, not overwritten with a real answer


@pytest.mark.asyncio
async def test_detects_ambiguous_query(monkeypatch):
    async def fake_call_llm_json(prompt_name, **kwargs):
        return _good_response(
            intent="other", complexity="medium", entities=[], keywords=[]
        )

    monkeypatch.setattr(
        "app.nodes.query_understanding.call_llm_json", fake_call_llm_json
    )

    state = {"question": "Tell me about it"}
    result = await query_understanding_node(state)
    assert result["attributes"]["is_ambiguous"] is True


@pytest.mark.asyncio
async def test_simple_trivial_query_not_flagged_ambiguous_even_with_no_entities(monkeypatch):
    """Edge case: 'hi' has no entities and might classify as intent=other,
    but complexity=simple means it's not actually ambiguous — it's trivial."""
    async def fake_call_llm_json(prompt_name, **kwargs):
        return _good_response(intent="other", complexity="simple", entities=[])

    monkeypatch.setattr(
        "app.nodes.query_understanding.call_llm_json", fake_call_llm_json
    )

    state = {"question": "hi"}
    result = await query_understanding_node(state)
    assert result["attributes"]["is_ambiguous"] is False


@pytest.mark.asyncio
async def test_raises_llm_call_error_on_schema_violation(monkeypatch):
    """Edge case: LLM returns an intent value outside our enum — must fail
    loudly here, not propagate a garbage string into the router."""
    async def fake_call_llm_json(prompt_name, **kwargs):
        return {
            "intent": "definitely_not_a_valid_intent",
            "complexity": "simple",
            "freshness_needed": False,
            "requires_multiple_sources": False,
        }

    monkeypatch.setattr(
        "app.nodes.query_understanding.call_llm_json", fake_call_llm_json
    )

    with pytest.raises(LLMCallError):
        await query_understanding_node({"question": "anything"})


@pytest.mark.asyncio
async def test_raises_llm_call_error_on_missing_required_field(monkeypatch):
    """Edge case: LLM drops 'complexity' entirely."""
    async def fake_call_llm_json(prompt_name, **kwargs):
        return {
            "intent": "factual",
            "freshness_needed": False,
            "requires_multiple_sources": False,
        }

    monkeypatch.setattr(
        "app.nodes.query_understanding.call_llm_json", fake_call_llm_json
    )

    with pytest.raises(LLMCallError):
        await query_understanding_node({"question": "anything"})


@pytest.mark.asyncio
async def test_propagates_llm_call_error_after_retries_exhausted(monkeypatch):
    """Edge case: underlying LLM call itself fails (network/provider error)
    after the client's own retry budget is exhausted."""
    async def failing_call_llm_json(prompt_name, **kwargs):
        raise LLMCallError("simulated provider outage")

    monkeypatch.setattr(
        "app.nodes.query_understanding.call_llm_json", failing_call_llm_json
    )

    with pytest.raises(LLMCallError):
        await query_understanding_node({"question": "anything"})


@pytest.mark.asyncio
async def test_whitespace_only_keywords_are_filtered(monkeypatch):
    async def fake_call_llm_json(prompt_name, **kwargs):
        return _good_response(keywords=["  ", "langgraph", "   "])

    monkeypatch.setattr(
        "app.nodes.query_understanding.call_llm_json", fake_call_llm_json
    )

    result = await query_understanding_node({"question": "What is LangGraph?"})
    assert result["attributes"]["keywords"] == ["langgraph"]


@pytest.mark.asyncio
async def test_empty_domain_defaults_to_general(monkeypatch):
    async def fake_call_llm_json(prompt_name, **kwargs):
        return _good_response(domain="   ")

    monkeypatch.setattr(
        "app.nodes.query_understanding.call_llm_json", fake_call_llm_json
    )

    result = await query_understanding_node({"question": "anything"})
    assert result["attributes"]["domain"] == "general"

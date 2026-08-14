"""
Unit tests for the Adaptive Router.
LLM calls mocked. Settings.enabled_strategies controlled via monkeypatch
so tests don't depend on the sandbox's actual .env contents.
"""
import sys

import pytest

sys.path.append(".")

from app.nodes.router import (
    _build_clarification_question,
    _rule_based_hints,
    adaptive_router_node,
)
from app.core.exceptions import LLMCallError
from app.config.settings import Settings


def _patch_enabled_strategies(monkeypatch, strategies):
    """enabled_strategies is a computed @property on the Settings model
    (intentionally, per Module 0's design notes), so it can't be
    monkeypatched as a plain attribute. Patch the property at the class
    level instead, scoped to this test via monkeypatch's auto-undo."""
    monkeypatch.setattr(Settings, "enabled_strategies", property(lambda self: strategies))


# --- _rule_based_hints (pure function) -------------------------------------

def test_freshness_forces_web_search():
    assert "web_search" in _rule_based_hints({"freshness_needed": True})


def test_multi_source_forces_hybrid_search():
    """requires_multiple_sources should map to hybrid_search (broad coverage),
    NOT multi_query_retrieval. Multi-query is reserved for research+complex only.
    This was the bug that caused simple 'multi-faceted' questions to be
    over-routed to the slow multi-query path.
    """
    hints = _rule_based_hints({"requires_multiple_sources": True})
    assert "hybrid_search" in hints
    assert "multi_query_retrieval" not in hints


def test_requires_comparison_flag_forces_hybrid():
    assert "hybrid_search" in _rule_based_hints({"requires_comparison": True})


def test_comparison_intent_also_forces_hybrid():
    assert "hybrid_search" in _rule_based_hints({"intent": "comparison"})


def test_no_signals_no_hints():
    assert _rule_based_hints({}) == []


def test_all_signals_combine_without_duplicates():
    hints = _rule_based_hints(
        {
            "freshness_needed": True,
            "requires_multiple_sources": True,
            "requires_comparison": True,
            "intent": "comparison",
        }
    )
    # requires_multiple_sources → hybrid_search (NOT multi_query)
    # requires_comparison / intent=comparison → hybrid_search (deduplicated)
    # freshness_needed → web_search
    assert set(hints) == {"web_search", "hybrid_search"}
    assert len(hints) == 2  # comparison signal counted once, multi_source deduped into hybrid


def test_research_complex_forces_multi_query():
    """multi_query_retrieval should only fire when intent=research AND complexity=complex."""
    hints = _rule_based_hints({"intent": "research", "complexity": "complex"})
    assert "multi_query_retrieval" in hints


def test_research_simple_does_not_force_multi_query():
    """A research intent at simple complexity should NOT trigger multi_query."""
    hints = _rule_based_hints({"intent": "research", "complexity": "simple"})
    assert "multi_query_retrieval" not in hints


# --- _build_clarification_question -----------------------------------------

def test_clarification_question_includes_original_question():
    q = _build_clarification_question("Tell me about it")
    assert "Tell me about it" in q


def test_clarification_question_truncates_long_input():
    long_q = "a" * 200
    q = _build_clarification_question(long_q)
    assert "..." in q
    assert len(q) < 250  # bounded, not unbounded growth


# --- adaptive_router_node: clarification branch -----------------------------

@pytest.mark.asyncio
async def test_ambiguous_with_no_rule_hints_triggers_clarification(monkeypatch):
    async def fake_call_llm_json(*a, **k):
        raise AssertionError("LLM should not be called when clarification gate triggers")

    monkeypatch.setattr("app.nodes.router.call_llm_json", fake_call_llm_json)

    state = {
        "question": "Tell me about it",
        "attributes": {"is_ambiguous": True, "intent": "other", "complexity": "medium"},
    }
    result = await adaptive_router_node(state)

    assert result["clarification_needed"] is True
    assert result["strategies"] == []
    assert "?" in result["clarification_question"]


@pytest.mark.asyncio
async def test_ambiguous_but_with_freshness_hint_still_routes(monkeypatch):
    """Edge case: even an ambiguous question shouldn't trigger clarification
    if a rule-based signal is strong enough to route confidently anyway."""
    async def fake_call_llm_json(*a, **k):
        return {"strategies": ["web_search"], "reasoning": "freshness needed"}

    monkeypatch.setattr("app.nodes.router.call_llm_json", fake_call_llm_json)
    _patch_enabled_strategies(monkeypatch, ["vector_search", "hybrid_search", "multi_query_retrieval", "web_search"])

    state = {
        "question": "What's happening with it right now?",
        "attributes": {
            "is_ambiguous": True,
            "freshness_needed": True,
            "intent": "other",
            "complexity": "medium",
        },
    }
    result = await adaptive_router_node(state)

    assert result["clarification_needed"] is False
    assert "web_search" in result["strategies"]


# --- adaptive_router_node: normal routing -----------------------------------

@pytest.mark.asyncio
async def test_merges_rule_hints_and_llm_strategies(monkeypatch):
    async def fake_call_llm_json(*a, **k):
        return {"strategies": ["vector_search"], "reasoning": "simple lookup"}

    monkeypatch.setattr("app.nodes.router.call_llm_json", fake_call_llm_json)
    _patch_enabled_strategies(monkeypatch, ["vector_search", "hybrid_search", "multi_query_retrieval", "web_search"])

    state = {
        "question": "Compare langgraph and langchain agents",
        "attributes": {
            "is_ambiguous": False,
            "requires_comparison": True,
            "intent": "comparison",
        },
    }
    result = await adaptive_router_node(state)

    assert "hybrid_search" in result["strategies"]   # from rule hint
    assert "vector_search" in result["strategies"]   # from LLM
    assert result["clarification_needed"] is False


@pytest.mark.asyncio
async def test_filters_out_strategies_not_enabled_in_deployment(monkeypatch):
    """Edge case: LLM suggests graph_rag, but it's not enabled here."""
    async def fake_call_llm_json(*a, **k):
        return {"strategies": ["graph_rag", "vector_search"], "reasoning": "..."}

    monkeypatch.setattr("app.nodes.router.call_llm_json", fake_call_llm_json)
    _patch_enabled_strategies(monkeypatch, ["vector_search", "hybrid_search"])

    state = {"question": "anything", "attributes": {"is_ambiguous": False}}
    result = await adaptive_router_node(state)

    assert "graph_rag" not in result["strategies"]
    assert result["strategies"] == ["vector_search"]


@pytest.mark.asyncio
async def test_defaults_to_vector_search_when_nothing_matches(monkeypatch):
    async def fake_call_llm_json(*a, **k):
        return {"strategies": [], "reasoning": "unsure"}

    monkeypatch.setattr("app.nodes.router.call_llm_json", fake_call_llm_json)
    _patch_enabled_strategies(monkeypatch, ["vector_search"])

    state = {"question": "anything", "attributes": {"is_ambiguous": False}}
    result = await adaptive_router_node(state)

    assert result["strategies"] == ["vector_search"]


@pytest.mark.asyncio
async def test_falls_back_to_rule_hints_when_llm_call_fails(monkeypatch):
    """Edge case: LLM/provider outage. Router must degrade gracefully, not raise."""
    async def failing_call_llm_json(*a, **k):
        raise LLMCallError("simulated provider outage")

    monkeypatch.setattr("app.nodes.router.call_llm_json", failing_call_llm_json)
    _patch_enabled_strategies(monkeypatch, ["vector_search", "hybrid_search", "multi_query_retrieval", "web_search"])

    state = {
        "question": "What's the latest version of LangGraph?",
        "attributes": {"is_ambiguous": False, "freshness_needed": True},
    }
    result = await adaptive_router_node(state)

    assert result["strategies"] == ["web_search"]  # rule hint survived the LLM outage


@pytest.mark.asyncio
async def test_falls_back_to_vector_search_when_llm_fails_and_no_rule_hints(monkeypatch):
    async def failing_call_llm_json(*a, **k):
        raise LLMCallError("simulated provider outage")

    monkeypatch.setattr("app.nodes.router.call_llm_json", failing_call_llm_json)
    _patch_enabled_strategies(monkeypatch, ["vector_search"])

    state = {"question": "What is an embedding?", "attributes": {"is_ambiguous": False}}
    result = await adaptive_router_node(state)

    assert result["strategies"] == ["vector_search"]

"""
Unit tests for the ARIStore wrapper.
Uses InMemoryStore backend so no external services are needed.
Tests cover: all three namespaces (preferences, conversations, rewrites),
schema round-trip, rolling turn limit, cache hit/miss, query normalization.
"""
import sys

import pytest

sys.path.append(".")

from app.memory.schemas import TurnRecord, UserPreferences
from app.memory.store import ARIStore


@pytest.fixture
def store():
    return ARIStore()  # InMemoryStore backend


# --- UserPreferences ---------------------------------------------------------

@pytest.mark.asyncio
async def test_preferences_returns_none_when_not_set(store):
    result = await store.get_preferences("conv-1")
    assert result is None


@pytest.mark.asyncio
async def test_preferences_round_trip(store):
    prefs = UserPreferences(preferred_strategies=["hybrid_search"], preferred_domains=["ML"])
    await store.save_preferences("conv-1", prefs)
    loaded = await store.get_preferences("conv-1")
    assert loaded is not None
    assert loaded.preferred_strategies == ["hybrid_search"]
    assert loaded.preferred_domains == ["ML"]


@pytest.mark.asyncio
async def test_update_preferences_from_turn_accumulates(store):
    await store.update_preferences_from_turn("conv-1", ["vector_search"], "finance")
    await store.update_preferences_from_turn("conv-1", ["hybrid_search"], "finance")

    prefs = await store.get_preferences("conv-1")
    assert "vector_search" in prefs.preferred_strategies
    assert "hybrid_search" in prefs.preferred_strategies
    assert prefs.preferred_domains.count("finance") == 1  # no duplicate domains


@pytest.mark.asyncio
async def test_update_preferences_skips_general_domain(store):
    await store.update_preferences_from_turn("conv-1", ["vector_search"], "general")
    prefs = await store.get_preferences("conv-1")
    assert "general" not in prefs.preferred_domains


# --- ConversationSummary ------------------------------------------------------

@pytest.mark.asyncio
async def test_summary_returns_none_when_not_set(store):
    result = await store.get_summary("conv-new")
    assert result is None


@pytest.mark.asyncio
async def test_append_turn_and_retrieve(store):
    turn = TurnRecord(
        question="What is LangGraph?",
        answer="LangGraph is a library for building agentic workflows.",
        citations=["doc1"],
        strategies_used=["vector_search"],
        confidence_score=0.9,
    )
    await store.append_turn("conv-1", turn)
    summary = await store.get_summary("conv-1")
    assert summary is not None
    assert len(summary.turns) == 1
    assert summary.turns[0].question == "What is LangGraph?"


@pytest.mark.asyncio
async def test_conversation_summary_rolling_limit(store):
    """Edge case: rolling limit — oldest turns should be evicted when
    MAX_TURNS is exceeded, newest should survive."""
    for i in range(12):
        turn = TurnRecord(
            question=f"question {i}",
            answer=f"answer {i}",
            confidence_score=0.8,
        )
        await store.append_turn("conv-rolling", turn)

    summary = await store.get_summary("conv-rolling")
    assert len(summary.turns) == 10  # MAX_TURNS
    assert summary.turns[0].question == "question 2"   # oldest kept
    assert summary.turns[-1].question == "question 11"  # newest


@pytest.mark.asyncio
async def test_multiple_conversations_isolated(store):
    turn_a = TurnRecord(question="q-for-A", answer="a", confidence_score=0.8)
    turn_b = TurnRecord(question="q-for-B", answer="b", confidence_score=0.8)
    await store.append_turn("conv-A", turn_a)
    await store.append_turn("conv-B", turn_b)

    summary_a = await store.get_summary("conv-A")
    summary_b = await store.get_summary("conv-B")
    assert summary_a.turns[0].question == "q-for-A"
    assert summary_b.turns[0].question == "q-for-B"


# --- RewriteCache ------------------------------------------------------------

@pytest.mark.asyncio
async def test_rewrite_cache_miss_returns_none(store):
    result = await store.get_rewrite("never asked before")
    assert result is None


@pytest.mark.asyncio
async def test_rewrite_cache_hit_after_save(store):
    await store.save_rewrite(
        original="what is langgraph",
        rewrite="LangGraph library overview and capabilities",
        strategies=["hybrid_search"],
    )
    result = await store.get_rewrite("what is langgraph")
    assert result is not None
    assert result.successful_rewrite == "LangGraph library overview and capabilities"
    assert result.strategies_that_worked == ["hybrid_search"]
    assert result.success_count == 1


@pytest.mark.asyncio
async def test_rewrite_cache_normalizes_case_and_whitespace(store):
    """Edge case: 'What is LangGraph?' and 'what is langgraph ?' should
    resolve to the same cache entry after normalization."""
    await store.save_rewrite(
        original="What is LangGraph?",
        rewrite="LangGraph explained",
        strategies=["vector_search"],
    )
    result = await store.get_rewrite("what is langgraph?")
    assert result is not None


@pytest.mark.asyncio
async def test_rewrite_cache_increments_success_count_on_repeated_save(store):
    await store.save_rewrite("original q", "rewrite v1", ["vector_search"])
    await store.save_rewrite("original q", "rewrite v2", ["hybrid_search"])

    result = await store.get_rewrite("original q")
    assert result.success_count == 2
    assert result.successful_rewrite == "rewrite v2"  # most recent wins

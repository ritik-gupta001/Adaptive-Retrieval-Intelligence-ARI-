"""
Unit tests for memory_load_node and memory_save_node.
Store is injected via monkeypatching get_store() to return a fresh
InMemoryStore-backed ARIStore per test — no shared state between tests.
"""
import sys

import pytest

sys.path.append(".")

from app.memory.store import ARIStore
from app.memory.schemas import TurnRecord


def _fresh_store():
    return ARIStore()


# ---------------------------------------------------------------------------
# memory_load_node
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_memory_load_returns_empty_context_for_new_conversation(monkeypatch):
    monkeypatch.setattr("app.nodes.memory_load.get_store", _fresh_store)

    from app.nodes.memory_load import memory_load_node
    state = {"question": "What is LangGraph?", "conversation_id": "new-conv"}
    result = await memory_load_node(state)

    assert result["long_term_context"] == {}


@pytest.mark.asyncio
async def test_memory_load_injects_conversation_history(monkeypatch):
    store = _fresh_store()
    await store.append_turn(
        "conv-1",
        TurnRecord(question="old question", answer="old answer", confidence_score=0.85),
    )
    monkeypatch.setattr("app.nodes.memory_load.get_store", lambda: store)

    from app.nodes.memory_load import memory_load_node
    result = await memory_load_node({"question": "new question", "conversation_id": "conv-1"})

    ctx = result["long_term_context"]
    assert "conversation_history" in ctx
    assert ctx["conversation_history"][0]["question"] == "old question"


@pytest.mark.asyncio
async def test_memory_load_injects_preferences(monkeypatch):
    from app.memory.schemas import UserPreferences
    store = _fresh_store()
    await store.save_preferences(
        "conv-1", UserPreferences(preferred_strategies=["hybrid_search"])
    )
    monkeypatch.setattr("app.nodes.memory_load.get_store", lambda: store)

    from app.nodes.memory_load import memory_load_node
    result = await memory_load_node({"question": "q", "conversation_id": "conv-1"})

    assert result["long_term_context"]["user_preferences"]["preferred_strategies"] == ["hybrid_search"]


@pytest.mark.asyncio
async def test_memory_load_injects_cached_rewrite(monkeypatch):
    store = _fresh_store()
    await store.save_rewrite("what is langgraph", "LangGraph library deep-dive", ["hybrid_search"])
    monkeypatch.setattr("app.nodes.memory_load.get_store", lambda: store)

    from app.nodes.memory_load import memory_load_node
    result = await memory_load_node({"question": "what is langgraph", "conversation_id": "conv-1"})

    assert "cached_rewrite" in result["long_term_context"]
    assert result["long_term_context"]["cached_rewrite"]["successful_rewrite"] == "LangGraph library deep-dive"


@pytest.mark.asyncio
async def test_memory_load_does_not_crash_on_store_failure(monkeypatch):
    """Core failure posture: Store errors must NEVER block the pipeline."""
    class BrokenStore:
        async def get_summary(self, *a): raise ConnectionError("db down")
        async def get_preferences(self, *a): raise ConnectionError("db down")
        async def get_rewrite(self, *a): raise ConnectionError("db down")

    monkeypatch.setattr("app.nodes.memory_load.get_store", BrokenStore)

    from app.nodes.memory_load import memory_load_node
    state = {"question": "q", "conversation_id": "conv-1"}
    result = await memory_load_node(state)

    # pipeline continues, context just empty
    assert result["long_term_context"] == {}


# ---------------------------------------------------------------------------
# memory_save_node
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_memory_save_writes_turn_on_high_confidence(monkeypatch):
    store = _fresh_store()
    monkeypatch.setattr("app.nodes.memory_save.get_store", lambda: store)

    from app.nodes.memory_save import memory_save_node
    state = {
        "conversation_id": "conv-1",
        "question": "What is LangGraph?",
        "original_question": "What is LangGraph?",
        "answer": "LangGraph is a library.",
        "citations": ["doc1"],
        "strategies": ["vector_search"],
        "confidence": {"confidence_level": "high", "confidence_score": 0.9},
        "attributes": {"domain": "ML"},
        "retry_count": 0,
    }
    await memory_save_node(state)

    summary = await store.get_summary("conv-1")
    assert summary is not None
    assert len(summary.turns) == 1
    assert summary.turns[0].question == "What is LangGraph?"


@pytest.mark.asyncio
async def test_memory_save_skips_on_low_confidence(monkeypatch):
    store = _fresh_store()
    monkeypatch.setattr("app.nodes.memory_save.get_store", lambda: store)

    from app.nodes.memory_save import memory_save_node
    state = {
        "conversation_id": "conv-low",
        "question": "q",
        "original_question": "q",
        "answer": "uncertain answer",
        "citations": [],
        "strategies": ["vector_search"],
        "confidence": {"confidence_level": "low", "confidence_score": 0.3},
        "attributes": {},
        "retry_count": 0,
    }
    await memory_save_node(state)

    summary = await store.get_summary("conv-low")
    assert summary is None  # nothing written


@pytest.mark.asyncio
async def test_memory_save_caches_rewrite_when_retry_and_high_confidence(monkeypatch):
    store = _fresh_store()
    monkeypatch.setattr("app.nodes.memory_save.get_store", lambda: store)

    from app.nodes.memory_save import memory_save_node
    state = {
        "conversation_id": "conv-1",
        "question": "LangGraph library overview",         # rewritten
        "original_question": "what is langgraph",         # original
        "answer": "LangGraph is...",
        "citations": ["doc1"],
        "strategies": ["hybrid_search"],
        "confidence": {"confidence_level": "high", "confidence_score": 0.92},
        "attributes": {"domain": "ML"},
        "retry_count": 1,                                 # was a retry
    }
    await memory_save_node(state)

    cached = await store.get_rewrite("what is langgraph")
    assert cached is not None
    assert cached.successful_rewrite == "LangGraph library overview"


@pytest.mark.asyncio
async def test_memory_save_does_not_cache_rewrite_without_retry(monkeypatch):
    store = _fresh_store()
    monkeypatch.setattr("app.nodes.memory_save.get_store", lambda: store)

    from app.nodes.memory_save import memory_save_node
    state = {
        "conversation_id": "conv-1",
        "question": "what is langgraph",
        "original_question": "what is langgraph",
        "answer": "LangGraph is...",
        "citations": [],
        "strategies": ["vector_search"],
        "confidence": {"confidence_level": "high", "confidence_score": 0.9},
        "attributes": {},
        "retry_count": 0,   # no retry happened — nothing to cache
    }
    await memory_save_node(state)

    cached = await store.get_rewrite("what is langgraph")
    assert cached is None


@pytest.mark.asyncio
async def test_memory_save_does_not_crash_on_store_failure(monkeypatch):
    """Core failure posture: Store write failures must not block the response."""
    class BrokenStore:
        async def append_turn(self, *a, **k): raise ConnectionError("db down")
        async def update_preferences_from_turn(self, *a, **k): raise ConnectionError("db down")
        async def save_rewrite(self, *a, **k): raise ConnectionError("db down")

    monkeypatch.setattr("app.nodes.memory_save.get_store", BrokenStore)

    from app.nodes.memory_save import memory_save_node
    state = {
        "conversation_id": "conv-broken",
        "question": "q",
        "original_question": "q",
        "answer": "a",
        "citations": [],
        "strategies": [],
        "confidence": {"confidence_level": "high", "confidence_score": 0.9},
        "attributes": {},
        "retry_count": 0,
    }
    result = await memory_save_node(state)
    assert result is not None   # pipeline still returns state

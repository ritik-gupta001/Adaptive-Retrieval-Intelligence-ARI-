"""
Tests for the orchestration nodes. The key behavior under test is failure
isolation in retrieve_node: one failing strategy shouldn't fail the whole
node when others succeed, but all-failing should raise.
"""
import sys

import pytest

sys.path.append(".")

from app.core.exceptions import RetrievalError
from app.nodes.document_merge import document_merge_node
from app.nodes.retrieve import retrieve_node


class _FakeRetriever:
    def __init__(self, name, docs=None, should_fail=False):
        self.name = name
        self._docs = docs or []
        self._should_fail = should_fail

    async def retrieve(self, query, k):
        if self._should_fail:
            raise RetrievalError(f"{self.name} failed")
        return self._docs


@pytest.mark.asyncio
async def test_retrieve_node_raises_with_no_strategies():
    with pytest.raises(RetrievalError):
        await retrieve_node({"question": "x", "strategies": []})


@pytest.mark.asyncio
async def test_retrieve_node_succeeds_when_one_strategy_fails_another_succeeds(monkeypatch):
    def fake_get_retriever(name):
        if name == "web_search":
            return _FakeRetriever("web_search", should_fail=True)
        return _FakeRetriever(
            "hybrid_search",
            docs=[{"content": "doc1", "score": 0.8, "strategy": "hybrid_search"}],
        )

    monkeypatch.setattr("app.nodes.retrieve.get_retriever", fake_get_retriever)

    state = {"question": "x", "strategies": ["hybrid_search", "web_search"]}
    result = await retrieve_node(state)

    assert len(result["retrieved_docs"]) == 1
    assert any("web_search" in entry for entry in result["issues_log"])


@pytest.mark.asyncio
async def test_retrieve_node_raises_when_all_strategies_fail(monkeypatch):
    def fake_get_retriever(name):
        return _FakeRetriever(name, should_fail=True)

    monkeypatch.setattr("app.nodes.retrieve.get_retriever", fake_get_retriever)

    state = {"question": "x", "strategies": ["vector_search", "web_search"]}
    with pytest.raises(RetrievalError):
        await retrieve_node(state)


@pytest.mark.asyncio
async def test_retrieve_node_combines_docs_from_multiple_successful_strategies(monkeypatch):
    def fake_get_retriever(name):
        return _FakeRetriever(
            name, docs=[{"content": f"doc from {name}", "score": 0.7, "strategy": name}]
        )

    monkeypatch.setattr("app.nodes.retrieve.get_retriever", fake_get_retriever)

    state = {"question": "x", "strategies": ["vector_search", "hybrid_search"]}
    result = await retrieve_node(state)

    assert len(result["retrieved_docs"]) == 2
    strategies_seen = {d["strategy"] for d in result["retrieved_docs"]}
    assert strategies_seen == {"vector_search", "hybrid_search"}


@pytest.mark.asyncio
async def test_document_merge_node_dedupes_across_strategies():
    state = {
        "retrieved_docs": [
            {"content": "Same chunk.", "score": 0.7, "strategy": "vector_search"},
            {"content": "Same chunk.", "score": 0.9, "strategy": "hybrid_search"},
            {"content": "Different chunk.", "score": 0.5, "strategy": "web_search"},
        ]
    }
    result = await document_merge_node(state)

    assert len(result["merged_docs"]) == 2
    # higher-scored duplicate copy (hybrid_search's 0.9) should be the one kept
    kept = [d for d in result["merged_docs"] if d["content"] == "Same chunk."][0]
    assert kept["score"] == 0.9
    assert kept["strategy"] == "hybrid_search"


@pytest.mark.asyncio
async def test_document_merge_node_handles_empty_input():
    result = await document_merge_node({"retrieved_docs": []})
    assert result["merged_docs"] == []

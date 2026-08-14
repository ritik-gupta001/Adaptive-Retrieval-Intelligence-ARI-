"""
Unit tests for Module 4. External libs (sentence_transformers, cohere)
mocked so these run fast/free without downloading model weights or hitting
a live API.
"""
import sys
from unittest.mock import MagicMock

import pytest

sys.path.append(".")

from app.core.exceptions import RerankError


# --- BGEReranker --------------------------------------------------------------

@pytest.mark.asyncio
async def test_bge_reranker_scores_and_sorts(monkeypatch):
    from app.rerankers.bge_reranker import BGEReranker

    reranker = BGEReranker()
    fake_model = MagicMock()
    fake_model.predict.return_value = [0.2, 0.9, 0.5]
    monkeypatch.setattr(reranker, "_get_model", lambda: fake_model)

    docs = [
        {"content": "doc A", "score": 0.5},
        {"content": "doc B", "score": 0.5},
        {"content": "doc C", "score": 0.5},
    ]
    result = await reranker.rerank("query", docs, top_n=3)

    assert [d["content"] for d in result] == ["doc B", "doc C", "doc A"]
    assert result[0]["score"] == 0.9


@pytest.mark.asyncio
async def test_bge_reranker_respects_top_n(monkeypatch):
    from app.rerankers.bge_reranker import BGEReranker

    reranker = BGEReranker()
    fake_model = MagicMock()
    fake_model.predict.return_value = [0.1, 0.9, 0.5, 0.3]
    monkeypatch.setattr(reranker, "_get_model", lambda: fake_model)

    docs = [{"content": f"doc{i}", "score": 0.5} for i in range(4)]
    result = await reranker.rerank("query", docs, top_n=2)
    assert len(result) == 2


@pytest.mark.asyncio
async def test_bge_reranker_empty_docs_returns_empty_without_model_call(monkeypatch):
    from app.rerankers.bge_reranker import BGEReranker

    reranker = BGEReranker()

    def fail_if_called():
        raise AssertionError("model should not load for empty docs")

    monkeypatch.setattr(reranker, "_get_model", fail_if_called)
    result = await reranker.rerank("query", [], top_n=5)
    assert result == []


@pytest.mark.asyncio
async def test_bge_reranker_raises_rerank_error_on_model_failure(monkeypatch):
    from app.rerankers.bge_reranker import BGEReranker

    reranker = BGEReranker()

    def boom():
        raise OSError("model weights not found")

    monkeypatch.setattr(reranker, "_get_model", boom)
    with pytest.raises(RerankError):
        await reranker.rerank("query", [{"content": "x", "score": 0.5}], top_n=1)


# --- CohereReranker -------------------------------------------------------------

@pytest.mark.asyncio
async def test_cohere_reranker_maps_results_back_to_docs(monkeypatch):
    from app.rerankers.cohere_reranker import CohereReranker

    reranker = CohereReranker()
    fake_result_1 = MagicMock(index=1, relevance_score=0.95)
    fake_result_0 = MagicMock(index=0, relevance_score=0.4)
    fake_response = MagicMock(results=[fake_result_1, fake_result_0])
    monkeypatch.setattr(reranker, "_rerank_sync", lambda q, d, n: fake_response)

    docs = [{"content": "doc A", "score": 0.5}, {"content": "doc B", "score": 0.5}]
    result = await reranker.rerank("query", docs, top_n=2)

    assert result[0]["content"] == "doc B"
    assert result[0]["score"] == 0.95
    assert result[1]["content"] == "doc A"


@pytest.mark.asyncio
async def test_cohere_reranker_raises_when_api_key_missing(monkeypatch):
    from app.rerankers.cohere_reranker import CohereReranker
    from app.config.settings import settings

    reranker = CohereReranker()
    monkeypatch.setattr(settings, "cohere_api_key", None)

    with pytest.raises(RerankError):
        await reranker.rerank("query", [{"content": "x", "score": 0.5}], top_n=1)


# --- rerank_node (provider switch + fallback) -----------------------------------

@pytest.mark.asyncio
async def test_rerank_node_skips_when_no_docs():
    from app.nodes.rerank import rerank_node

    result = await rerank_node({"question": "q", "merged_docs": []})
    assert result["reranked_docs"] == []


@pytest.mark.asyncio
async def test_rerank_node_uses_configured_provider(monkeypatch):
    from app.nodes import rerank as rerank_module
    from app.config.settings import settings

    monkeypatch.setattr(settings, "reranker_provider", "bge")
    rerank_module._RERANKERS.clear()

    class FakeReranker:
        async def rerank(self, query, docs, top_n):
            return [dict(d, score=1.0) for d in docs][:top_n]

    monkeypatch.setattr(rerank_module, "_get_reranker", lambda: FakeReranker())

    state = {
        "question": "q",
        "merged_docs": [{"content": "a", "score": 0.5}, {"content": "b", "score": 0.6}],
    }
    result = await rerank_module.rerank_node(state)
    assert all(d["score"] == 1.0 for d in result["reranked_docs"])


@pytest.mark.asyncio
async def test_rerank_node_falls_back_to_merge_order_on_failure(monkeypatch):
    """Core reliability behavior for this module: reranker failure must
    not fail the request — must degrade to the pre-rerank order."""
    from app.nodes import rerank as rerank_module

    class FailingReranker:
        async def rerank(self, query, docs, top_n):
            raise RerankError("model failed to load")

    from app.config.settings import settings

    monkeypatch.setattr(settings, "reranker_provider", "bge")
    monkeypatch.setattr(rerank_module, "_get_reranker", lambda: FailingReranker())



    state = {
        "question": "q",
        "merged_docs": [
            {"content": "a", "score": 0.9},
            {"content": "b", "score": 0.7},
            {"content": "c", "score": 0.5},
        ],
    }
    result = await rerank_module.rerank_node(state)

    assert result["reranked_docs"] == state["merged_docs"]
    assert any("rerank_fallback" in entry for entry in result["issues_log"])


@pytest.mark.asyncio
async def test_rerank_node_fallback_respects_top_k(monkeypatch):
    from app.nodes import rerank as rerank_module
    from app.config.settings import settings

    monkeypatch.setattr(settings, "top_k", 2)

    class FailingReranker:
        async def rerank(self, query, docs, top_n):
            raise RerankError("api down")

    monkeypatch.setattr(rerank_module, "_get_reranker", lambda: FailingReranker())

    state = {
        "question": "q",
        "merged_docs": [{"content": str(i), "score": 1.0 - i * 0.1} for i in range(5)],
    }
    result = await rerank_module.rerank_node(state)
    assert len(result["reranked_docs"]) == 2

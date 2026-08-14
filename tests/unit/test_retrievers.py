"""
Unit tests for each retrieval strategy. External clients (chromadb,
rank_bm25, tavily, neo4j) are mocked so these run fast/free without real
services. Real-service integration is covered separately for strategies
where it's cheap to verify (see test_retrieve_and_merge_nodes.py's
registry-level test) — full live integration tests against a real Chroma/
Neo4j/Tavily instance are deferred to the Evaluation module, where they
belong alongside the golden-query harness.
"""
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.append(".")

from app.core.exceptions import RetrievalError


# --- VectorSearchRetriever ---------------------------------------------------

@pytest.mark.asyncio
async def test_vector_search_converts_distance_to_score(monkeypatch):
    from app.retrievers.vector_search import VectorSearchRetriever

    fake_collection = MagicMock()
    fake_collection.query.return_value = {
        "ids": [["id1", "id2"]],
        "documents": [["doc one", "doc two"]],
        "distances": [[0.1, 0.4]],
        "metadatas": [[{"source": "a.txt"}, {"source": "b.txt"}]],
    }

    retriever = VectorSearchRetriever()
    monkeypatch.setattr(retriever, "_get_collection", lambda: fake_collection)

    docs = await retriever.retrieve("query", k=2)

    assert len(docs) == 2
    assert docs[0]["score"] == pytest.approx(0.9)
    assert docs[1]["score"] == pytest.approx(0.6)
    assert all(d["strategy"] == "vector_search" for d in docs)


@pytest.mark.asyncio
async def test_vector_search_raises_retrieval_error_on_backend_failure(monkeypatch):
    from app.retrievers.vector_search import VectorSearchRetriever

    def boom():
        raise ConnectionError("chroma unreachable")

    retriever = VectorSearchRetriever()
    monkeypatch.setattr(retriever, "_get_collection", boom)

    with pytest.raises(RetrievalError):
        await retriever.retrieve("query", k=2)


# --- HybridSearchRetriever (RRF fusion) --------------------------------------

@pytest.mark.asyncio
async def test_hybrid_search_fuses_both_legs(monkeypatch):
    from app.retrievers.hybrid_search import HybridSearchRetriever

    retriever = HybridSearchRetriever()

    async def fake_vector_retrieve(query, k):
        return [
            {"content": "doc A", "source": "a", "score": 0.9, "strategy": "vector_search"},
            {"content": "doc B", "source": "b", "score": 0.5, "strategy": "vector_search"},
        ]

    def fake_bm25_search(query, k):
        return [
            {"content": "doc B", "source": "b", "score": 0.95},
            {"content": "doc C", "source": "c", "score": 0.6},
        ]

    monkeypatch.setattr(retriever._vector, "retrieve", fake_vector_retrieve)
    monkeypatch.setattr(retriever._bm25, "search", fake_bm25_search)

    docs = await retriever.retrieve("query", k=3)

    contents = [d["content"] for d in docs]
    assert "doc B" in contents  # appears in both legs -> should rank highly
    assert docs[0]["content"] == "doc B"  # RRF should favor the doc both legs agree on
    assert all(d["strategy"] == "hybrid_search" for d in docs)


@pytest.mark.asyncio
async def test_hybrid_search_degrades_when_one_leg_fails(monkeypatch):
    from app.retrievers.hybrid_search import HybridSearchRetriever

    retriever = HybridSearchRetriever()

    async def fake_vector_retrieve(query, k):
        return [{"content": "doc A", "source": "a", "score": 0.9, "strategy": "vector_search"}]

    def failing_bm25(query, k):
        raise RetrievalError("bm25 corpus missing")

    monkeypatch.setattr(retriever._vector, "retrieve", fake_vector_retrieve)
    monkeypatch.setattr(retriever._bm25, "search", failing_bm25)

    docs = await retriever.retrieve("query", k=3)
    assert len(docs) == 1
    assert docs[0]["content"] == "doc A"


@pytest.mark.asyncio
async def test_hybrid_search_raises_when_both_legs_fail(monkeypatch):
    from app.retrievers.hybrid_search import HybridSearchRetriever

    retriever = HybridSearchRetriever()

    async def failing_vector(query, k):
        raise RetrievalError("chroma down")

    def failing_bm25(query, k):
        raise RetrievalError("bm25 corpus missing")

    monkeypatch.setattr(retriever._vector, "retrieve", failing_vector)
    monkeypatch.setattr(retriever._bm25, "search", failing_bm25)

    with pytest.raises(RetrievalError):
        await retriever.retrieve("query", k=3)


def test_rrf_fuse_pure_math():
    from app.retrievers.hybrid_search import HybridSearchRetriever

    list_a = [{"content": "x", "score": 1.0}, {"content": "y", "score": 0.9}]
    list_b = [{"content": "y", "score": 1.0}, {"content": "z", "score": 0.9}]

    fused = HybridSearchRetriever._rrf_fuse([list_a, list_b], k=3, weights=[0.6, 0.4])
    contents = [d["content"] for d in fused]
    # "y" appears rank-1 in list_b and rank-2 in list_a -> should outscore
    # "x" (rank-1 in list_a only) given comparable weights.
    assert contents[0] == "y"


# --- MultiQueryRetriever ------------------------------------------------------

@pytest.mark.asyncio
async def test_multi_query_generates_and_merges(monkeypatch):
    from app.retrievers.multi_query import MultiQueryRetriever

    retriever = MultiQueryRetriever()

    async def fake_generate_queries(question):
        return ["sub query 1", "sub query 2"]

    fake_results = {
        "sub query 1": [{"content": "LangGraph checkpointing saves state per step.", "source": "x", "score": 0.8}],
        "sub query 2": [{"content": "The Store API persists long-term memory across threads.", "source": "y", "score": 0.75}],
    }

    async def fake_base_retrieve(query, k):
        return fake_results[query]

    monkeypatch.setattr(retriever, "_generate_queries", fake_generate_queries)
    monkeypatch.setattr(retriever._base_retriever, "retrieve", fake_base_retrieve)

    docs = await retriever.retrieve("original question", k=5)
    assert len(docs) == 2  # genuinely distinct content from each sub-query, both kept


@pytest.mark.asyncio
async def test_multi_query_dedupes_near_duplicate_results_across_sub_queries(monkeypatch):
    """Edge case: two sub-queries (e.g. 'LangGraph persistence' and
    'LangGraph checkpointing') legitimately can surface the same underlying
    chunk. This must collapse to one result, not be double-counted."""
    from app.retrievers.multi_query import MultiQueryRetriever

    retriever = MultiQueryRetriever()

    async def fake_generate_queries(question):
        return ["LangGraph persistence", "LangGraph checkpointing"]

    async def fake_base_retrieve(query, k):
        # both sub-queries happen to surface the same chunk, slightly
        # reworded the way two near-identical retrievals often are
        return [{"content": "LangGraph persists state via checkpoints.", "source": "doc1", "score": 0.85}]

    monkeypatch.setattr(retriever, "_generate_queries", fake_generate_queries)
    monkeypatch.setattr(retriever._base_retriever, "retrieve", fake_base_retrieve)

    docs = await retriever.retrieve("LangGraph persistence", k=5)
    assert len(docs) == 1
    assert all(d["strategy"] == "multi_query_retrieval" for d in docs)


@pytest.mark.asyncio
async def test_multi_query_falls_back_to_original_question_on_generation_failure(monkeypatch):
    from app.retrievers.multi_query import MultiQueryRetriever
    from app.core.exceptions import LLMCallError
    from app.core.llm import call_llm_json as real_call_llm_json  # noqa: F401

    retriever = MultiQueryRetriever()

    async def failing_call_llm_json(*a, **k):
        raise LLMCallError("simulated outage")

    monkeypatch.setattr("app.retrievers.multi_query.call_llm_json", failing_call_llm_json)

    async def fake_base_retrieve(query, k):
        assert query == "original question"  # proves fallback used the original
        return [{"content": "fallback result", "source": "x", "score": 0.7}]

    monkeypatch.setattr(retriever._base_retriever, "retrieve", fake_base_retrieve)

    docs = await retriever.retrieve("original question", k=5)
    assert len(docs) == 1


@pytest.mark.asyncio
async def test_multi_query_raises_when_all_sub_queries_fail(monkeypatch):
    from app.retrievers.multi_query import MultiQueryRetriever

    retriever = MultiQueryRetriever()

    async def fake_generate_queries(question):
        return ["q1", "q2"]

    async def failing_retrieve(query, k):
        raise RetrievalError("vector store down")

    monkeypatch.setattr(retriever, "_generate_queries", fake_generate_queries)
    monkeypatch.setattr(retriever._base_retriever, "retrieve", failing_retrieve)

    with pytest.raises(RetrievalError):
        await retriever.retrieve("question", k=5)


# --- WebSearchRetriever -------------------------------------------------------

@pytest.mark.asyncio
async def test_web_search_parses_tavily_response(monkeypatch):
    from app.retrievers.web_search import WebSearchRetriever

    fake_client = MagicMock()
    fake_client.search.return_value = {
        "results": [
            {"content": "latest news content", "url": "http://x.com", "title": "X", "score": 0.88},
        ]
    }

    retriever = WebSearchRetriever()
    monkeypatch.setattr(retriever, "_get_client", lambda: fake_client)

    docs = await retriever.retrieve("latest news", k=3)
    assert docs[0]["source"] == "http://x.com"
    assert docs[0]["score"] == 0.88
    assert docs[0]["strategy"] == "web_search"


@pytest.mark.asyncio
async def test_web_search_raises_when_api_key_missing(monkeypatch):
    from app.retrievers.web_search import WebSearchRetriever
    from app.config.settings import settings

    retriever = WebSearchRetriever()
    monkeypatch.setattr(settings, "tavily_api_key", None)

    with pytest.raises(RetrievalError):
        await retriever.retrieve("query", k=3)


# --- GraphRAGRetriever ---------------------------------------------------------

@pytest.mark.asyncio
async def test_graph_rag_raises_clearly_when_disabled(monkeypatch):
    from app.retrievers.graph_rag import GraphRAGRetriever
    from app.config.settings import settings

    retriever = GraphRAGRetriever()
    monkeypatch.setattr(settings, "graph_rag_enabled", False)

    with pytest.raises(RetrievalError, match="GRAPH_RAG_ENABLED"):
        await retriever.retrieve("query", k=3)


# --- BaseRetriever Helpers -----------------------------------------------------

def test_parse_chromadb_results_empty_or_none():
    from app.retrievers.base import parse_chromadb_results
    assert parse_chromadb_results(None) == []
    assert parse_chromadb_results({}) == []


def test_parse_chromadb_results_valid():
    from app.retrievers.base import parse_chromadb_results
    raw = {
        "ids": [["id1"]],
        "documents": [["hello world"]],
        "distances": [[0.2]],
        "metadatas": [[{"source": "test.txt"}]],
    }
    docs = parse_chromadb_results(raw)
    assert len(docs) == 1
    assert docs[0]["content"] == "hello world"
    assert docs[0]["source"] == "test.txt"
    assert docs[0]["score"] == pytest.approx(0.8)


def test_get_uploaded_sources(tmp_path):
    from app.retrievers.base import get_uploaded_sources
    import json

    corpus_file = tmp_path / "bm25_corpus.json"
    corpus_data = [
        {"content": "c1", "source": "data/raw/doc1.txt"},
        {"content": "c2", "source": "uploaded_file.pdf"},
    ]
    corpus_file.write_text(json.dumps(corpus_data), encoding="utf-8")

    sources = get_uploaded_sources(corpus_file)
    assert sources == ["uploaded_file.pdf"]


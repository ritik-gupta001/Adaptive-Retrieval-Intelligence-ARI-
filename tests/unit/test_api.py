"""
API tests for Module 11.
Graph invocation is mocked via monkeypatching get_compiled_graph() at the
module where it is actually called (app.api.query / app.api.stream /
app.api.health). The lifespan degrades gracefully on missing graph, so
we don't need to stub it for the TestClient fixture.
"""
import json
import sys
from unittest.mock import MagicMock

import pytest

sys.path.append(".")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fake_state():
    return {
        "answer": "LangGraph is a library for building agentic workflows.",
        "citations": ["docs/langgraph.md"],
        "confidence": {
            "confidence_score": 0.88,
            "hallucination_risk": 0.02,
            "retrieval_quality": 0.9,
            "reflection_score": 0.85,
            "citation_quality": 0.9,
            "num_sources": 1,
            "confidence_level": "high",
            "reason": "All sub-scores within acceptable range.",
        },
        "strategies": ["vector_search"],
        "retry_count": 0,
        "issues_log": [],
        "strategy_reasoning": "simple factual",
        "clarification_needed": False,
        "clarification_question": None,
    }


def _make_client(monkeypatch, fake_graph=None):
    """Creates a TestClient with graph mocked at the query module level."""
    from fastapi.testclient import TestClient
    from app.main import create_app

    if fake_graph is None:
        fake_graph = MagicMock()
        fake_graph.invoke.return_value = _fake_state()

    monkeypatch.setattr("app.api.query.get_compiled_graph", lambda: fake_graph)
    return TestClient(create_app(), raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

def test_root_endpoint():
    from fastapi.testclient import TestClient
    from app.main import create_app

    c = TestClient(create_app(), raise_server_exceptions=False)
    resp = c.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "online"
    assert "health" in data


def test_health_returns_ok_or_degraded():
    from fastapi.testclient import TestClient
    from app.main import create_app
    c = TestClient(create_app(), raise_server_exceptions=False)
    resp = c.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("ok", "degraded")
    assert "timestamp" in data
    assert "checkpointer_backend" in data


# ---------------------------------------------------------------------------
# GET /graph
# ---------------------------------------------------------------------------

def test_graph_structure_endpoint(monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import create_app

    fake_edge = MagicMock()
    fake_edge.source = "query_understanding"
    fake_edge.target = "adaptive_router"

    fake_g = MagicMock()
    fake_g.nodes.keys.return_value = [
        "__start__", "query_understanding", "adaptive_router", "__end__"
    ]
    fake_g.edges = [fake_edge]

    fake_compiled = MagicMock()
    fake_compiled.get_graph.return_value = fake_g

    monkeypatch.setattr(
        "app.api.health.build_graph",
        lambda: MagicMock(compile=lambda: fake_compiled)
    )

    c = TestClient(create_app(), raise_server_exceptions=False)
    resp = c.get("/graph")
    assert resp.status_code == 200
    body = resp.json()
    assert "nodes" in body
    assert "edges" in body


# ---------------------------------------------------------------------------
# POST /query — success path
# ---------------------------------------------------------------------------

def test_query_returns_answer_and_citations(monkeypatch):
    c = _make_client(monkeypatch)
    resp = c.post("/query", json={"question": "What is LangGraph?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "LangGraph" in data["answer"]
    assert data["citations"] == ["docs/langgraph.md"]


def test_query_returns_full_confidence_detail(monkeypatch):
    c = _make_client(monkeypatch)
    resp = c.post("/query", json={"question": "What is LangGraph?"})
    data = resp.json()
    assert data["confidence"]["confidence_level"] == "high"
    assert data["confidence"]["confidence_score"] == pytest.approx(0.88)
    assert data["confidence"]["num_sources"] == 1


def test_query_generates_conversation_id_if_not_provided(monkeypatch):
    c = _make_client(monkeypatch)
    resp = c.post("/query", json={"question": "What is LangGraph?"})
    data = resp.json()
    assert data["conversation_id"]
    assert data["thread_id"]


def test_query_preserves_provided_conversation_id(monkeypatch):
    c = _make_client(monkeypatch)
    resp = c.post(
        "/query",
        json={"question": "What is LangGraph?", "conversation_id": "test-conv-123"},
    )
    assert resp.json()["conversation_id"] == "test-conv-123"


def test_query_passes_thread_id_in_graph_config(monkeypatch):
    """thread_id must appear in the RunnableConfig so the checkpointer
    knows which thread to snapshot — this is the key multi-turn wire."""
    captured = {}
    fake_graph = MagicMock()

    def capture_invoke(state, config):
        captured.update(config)
        return _fake_state()

    fake_graph.invoke.side_effect = capture_invoke
    c = _make_client(monkeypatch, fake_graph=fake_graph)
    c.post("/query", json={"question": "q", "thread_id": "my-thread-xyz"})

    assert captured.get("configurable", {}).get("thread_id") == "my-thread-xyz"


def test_query_thread_id_defaults_to_conversation_id_when_not_provided(monkeypatch):
    captured = {}
    fake_graph = MagicMock()

    def capture_invoke(state, config):
        captured.update(config)
        return _fake_state()

    fake_graph.invoke.side_effect = capture_invoke
    c = _make_client(monkeypatch, fake_graph=fake_graph)
    c.post("/query", json={"question": "q", "conversation_id": "conv-abc"})

    assert captured["configurable"]["thread_id"] == "conv-abc"


# ---------------------------------------------------------------------------
# POST /query — error handling
# ---------------------------------------------------------------------------

def test_query_returns_500_on_unexpected_error(monkeypatch):
    fake_graph = MagicMock()
    fake_graph.invoke.side_effect = RuntimeError("unexpected error")
    c = _make_client(monkeypatch, fake_graph=fake_graph)
    resp = c.post("/query", json={"question": "q"})
    assert resp.status_code == 500


def test_query_rejects_empty_question(monkeypatch):
    c = _make_client(monkeypatch)
    resp = c.post("/query", json={"question": ""})
    assert resp.status_code == 422


def test_query_rejects_missing_question_field(monkeypatch):
    c = _make_client(monkeypatch)
    resp = c.post("/query", json={})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /query/stream — SSE format
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_emits_node_complete_and_final_events(monkeypatch):
    import httpx
    from app.main import create_app

    async def fake_astream_events(state, config, version="v2"):
        for node in ["query_understanding", "adaptive_router", "generate", "finalize"]:
            yield {
                "event": "on_chain_end",
                "name": node,
                "data": {"output": _fake_state()},
            }

    fake_graph = MagicMock()
    fake_graph.astream_events = fake_astream_events
    monkeypatch.setattr("app.api.stream.get_compiled_graph", lambda: fake_graph)

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        async with ac.stream("POST", "/query/stream", json={"question": "q"}) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]

            events = []
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))

    event_types = [e["event"] for e in events]
    assert "node_complete" in event_types
    assert "final" in event_types


@pytest.mark.asyncio
async def test_stream_node_complete_includes_label(monkeypatch):
    import httpx
    from app.main import create_app

    async def fake_astream_events(state, config, version="v2"):
        yield {
            "event": "on_chain_end",
            "name": "query_understanding",
            "data": {"output": _fake_state()},
        }

    fake_graph = MagicMock()
    fake_graph.astream_events = fake_astream_events
    monkeypatch.setattr("app.api.stream.get_compiled_graph", lambda: fake_graph)

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        async with ac.stream("POST", "/query/stream", json={"question": "q"}) as resp:
            events = []
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))

    node_events = [e for e in events if e.get("event") == "node_complete"]
    assert any(e.get("label") == "Understanding query" for e in node_events)


@pytest.mark.asyncio
async def test_stream_emits_error_event_on_graph_failure(monkeypatch):
    import httpx
    from app.main import create_app

    async def failing_astream_events(state, config, version="v2"):
        raise RuntimeError("graph exploded")
        yield  # makes it an async generator

    fake_graph = MagicMock()
    fake_graph.astream_events = failing_astream_events
    monkeypatch.setattr("app.api.stream.get_compiled_graph", lambda: fake_graph)

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        async with ac.stream("POST", "/query/stream", json={"question": "q"}) as resp:
            events = []
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))

    assert any(e["event"] == "error" for e in events)

"""
Unit tests for Module 12 observability.

Layer 1 (LangSmith auto-tracing) has no unit tests — it's env-var config
with no logic to test.
Layer 2 (tag_run) tests are skipped — they require a live LangSmith client
and belong in integration tests once we have a test project key.
Layer 3 (extract_run_metrics) is pure and gets thorough coverage.
Middleware is tested via its effect on response headers.
"""
import sys

import pytest

sys.path.append(".")

from app.observability.metrics import RunMetrics, extract_run_metrics


# ---------------------------------------------------------------------------
# extract_run_metrics — pure function, no mocks needed
# ---------------------------------------------------------------------------

def _full_state():
    return {
        "run_id": "run-abc-123",
        "question": "What is LangGraph?",
        "original_question": "What is LangGraph?",
        "strategies": ["hybrid_search"],
        "strategy_reasoning": "comparison intent detected",
        "retry_count": 1,
        "retrieved_docs": [{"content": "a"}, {"content": "b"}, {"content": "c"}],
        "merged_docs": [{"content": "a"}, {"content": "b"}],
        "reranked_docs": [{"content": "a"}],
        "validation": {
            "recommendation": "proceed",
            "relevance_score": 0.85,
        },
        "reflection": {
            "is_supported": True,
            "hallucinations": [],
            "overall_score": 0.9,
        },
        "confidence": {
            "confidence_score": 0.87,
            "confidence_level": "high",
            "hallucination_risk": 0.02,
            "retrieval_quality": 0.85,
            "citation_quality": 0.9,
        },
        "citations": ["doc1.md", "doc2.md"],
        "issues_log": ["validation:minor coverage gap"],
        "clarification_needed": False,
        "long_term_context": {"conversation_history": [{"question": "prev q"}]},
    }


def test_extract_run_metrics_full_state():
    metrics = extract_run_metrics(_full_state())

    assert metrics.run_id == "run-abc-123"
    assert metrics.question == "What is LangGraph?"
    assert metrics.strategies_used == ["hybrid_search"]
    assert metrics.retry_count == 1
    assert metrics.docs_retrieved == 3
    assert metrics.docs_after_merge == 2
    assert metrics.docs_after_rerank == 1
    assert metrics.validation_recommendation == "proceed"
    assert metrics.validation_relevance_score == pytest.approx(0.85)
    assert metrics.reflection_is_supported is True
    assert metrics.reflection_hallucinations == []
    assert metrics.reflection_overall_score == pytest.approx(0.9)
    assert metrics.confidence_score == pytest.approx(0.87)
    assert metrics.confidence_level == "high"
    assert metrics.hallucination_risk == pytest.approx(0.02)
    assert metrics.num_citations == 2
    assert metrics.had_memory_context is True
    assert metrics.clarification_needed is False


def test_extract_run_metrics_empty_state_has_safe_defaults():
    """Edge case: graph short-circuited early (e.g. clarification path)
    leaves most fields unpopulated — extract must not raise."""
    metrics = extract_run_metrics({})

    assert metrics.run_id == "unknown"
    assert metrics.question == ""
    assert metrics.strategies_used == []
    assert metrics.docs_retrieved == 0
    assert metrics.confidence_score == 0.0
    assert metrics.confidence_level == "low"
    assert metrics.had_memory_context is False
    assert metrics.clarification_needed is False


def test_extract_run_metrics_clarification_path():
    state = {
        "run_id": "run-clarify",
        "question": "Tell me about it",
        "original_question": "Tell me about it",
        "clarification_needed": True,
        "clarification_question": "What do you mean by 'it'?",
        "strategies": [],
        "retry_count": 0,
    }
    metrics = extract_run_metrics(state)
    assert metrics.clarification_needed is True
    assert metrics.docs_retrieved == 0
    assert metrics.strategies_used == []


def test_extract_uses_original_question_not_rewritten():
    """Edge case: after a rewrite loop, state.question is the rewritten
    version. Metrics should always report the original question."""
    state = {
        "question": "LangGraph library deep-dive",      # rewritten
        "original_question": "what is langgraph",       # original
    }
    metrics = extract_run_metrics(state)
    assert metrics.question == "what is langgraph"


def test_extract_handles_none_nested_fields():
    """Edge case: fields that are explicitly set to None (by query_rewrite's
    stale field clearing) should fall back to defaults, not crash."""
    state = {
        "run_id": "run-1",
        "question": "q",
        "confidence": None,
        "reflection": None,
        "validation": None,
        "retrieved_docs": None,
        "merged_docs": None,
        "reranked_docs": None,
        "citations": None,
        "issues_log": None,
        "long_term_context": None,
    }
    metrics = extract_run_metrics(state)
    assert metrics.confidence_score == 0.0
    assert metrics.docs_retrieved == 0
    assert metrics.num_citations == 0


# ---------------------------------------------------------------------------
# RunMetrics.is_high_quality
# ---------------------------------------------------------------------------

def test_is_high_quality_true_for_clean_high_confidence_run():
    metrics = extract_run_metrics({
        **_full_state(),
        "confidence": {
            "confidence_score": 0.9,
            "confidence_level": "high",
            "hallucination_risk": 0.0,
            "retrieval_quality": 0.9,
            "citation_quality": 0.9,
        },
        "reflection": {
            "is_supported": True,
            "hallucinations": [],
            "overall_score": 0.9,
        },
    })
    assert metrics.is_high_quality is True


def test_is_high_quality_false_when_hallucinations_present():
    state = _full_state()
    state["reflection"]["hallucinations"] = ["fabricated claim"]
    metrics = extract_run_metrics(state)
    assert metrics.is_high_quality is False


def test_is_high_quality_false_when_low_confidence():
    state = _full_state()
    state["confidence"]["confidence_level"] = "low"
    metrics = extract_run_metrics(state)
    assert metrics.is_high_quality is False


# ---------------------------------------------------------------------------
# RunMetrics.to_dict — serialization
# ---------------------------------------------------------------------------

def test_to_dict_contains_all_expected_keys():
    metrics = extract_run_metrics(_full_state())
    d = metrics.to_dict()

    required_keys = {
        "run_id", "question", "strategies_used", "strategy_reasoning",
        "retry_count", "docs_retrieved", "docs_after_merge", "docs_after_rerank",
        "validation_recommendation", "validation_relevance_score",
        "reflection_is_supported", "reflection_hallucinations",
        "reflection_overall_score", "confidence_score", "confidence_level",
        "hallucination_risk", "retrieval_quality", "citation_quality",
        "num_citations", "issues_log", "clarification_needed", "had_memory_context",
    }
    assert required_keys.issubset(set(d.keys()))


def test_to_dict_is_json_serializable():
    import json
    metrics = extract_run_metrics(_full_state())
    # Should not raise
    serialized = json.dumps(metrics.to_dict())
    assert "run_id" in serialized


# ---------------------------------------------------------------------------
# ObservabilityMiddleware — header injection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_middleware_injects_run_id_and_latency_headers():
    import httpx
    from app.main import create_app

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/health")

    assert "x-run-id" in resp.headers
    assert "x-response-time-ms" in resp.headers
    # run_id should be a non-empty string
    assert len(resp.headers["x-run-id"]) > 0
    # latency should be parseable as a float
    float(resp.headers["x-response-time-ms"])


@pytest.mark.asyncio
async def test_middleware_unique_run_id_per_request():
    import httpx
    from app.main import create_app

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp1 = await ac.get("/health")
        resp2 = await ac.get("/health")

    assert resp1.headers["x-run-id"] != resp2.headers["x-run-id"]

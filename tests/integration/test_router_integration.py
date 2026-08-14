
import sys

import pytest

sys.path.append(".")

from app.config.settings import settings
from app.nodes.router import adaptive_router_node

pytestmark = pytest.mark.asyncio

_has_key = bool(settings.anthropic_api_key or settings.openai_api_key)


@pytest.mark.skipif(not _has_key, reason="No LLM API key configured — skipping live integration test")
async def test_real_router_picks_vector_search_for_simple_question():
    state = {
        "question": "What is the capital of France?",
        "attributes": {
            "is_ambiguous": False,
            "intent": "factual",
            "complexity": "simple",
            "freshness_needed": False,
            "requires_multiple_sources": False,
        },
    }
    result = await adaptive_router_node(state)
    assert "vector_search" in result["strategies"]
    assert result["clarification_needed"] is False


@pytest.mark.skipif(not _has_key, reason="No LLM API key configured — skipping live integration test")
async def test_real_router_routes_comparison_to_hybrid_search():
    state = {
        "question": "Compare hybrid search and vector search for retrieval quality.",
        "attributes": {
            "is_ambiguous": False,
            "intent": "comparison",
            "complexity": "medium",
            "requires_comparison": True,
            "freshness_needed": False,
        },
    }
    result = await adaptive_router_node(state)
    assert "hybrid_search" in result["strategies"]

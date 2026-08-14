
import sys

import pytest

sys.path.append(".")

from app.config.settings import settings
from app.nodes.query_understanding import query_understanding_node

pytestmark = pytest.mark.asyncio

_has_key = bool(settings.anthropic_api_key or settings.openai_api_key)


@pytest.mark.skipif(not _has_key, reason="No LLM API key configured — skipping live integration test")
async def test_real_llm_classifies_simple_question():
    state = {"question": "What is the capital of France?"}
    result = await query_understanding_node(state)

    attrs = result["attributes"]
    assert attrs["intent"] in ("factual", "other")
    assert attrs["complexity"] == "simple"
    assert attrs["freshness_needed"] is False
    # the hard rule, checked end-to-end against a real model
    assert "answer" not in result or result.get("answer", "") == ""


@pytest.mark.skipif(not _has_key, reason="No LLM API key configured — skipping live integration test")
async def test_real_llm_flags_freshness_for_latest_news_question():
    state = {"question": "What's the latest news on interest rate decisions this week?"}
    result = await query_understanding_node(state)

    assert result["attributes"]["freshness_needed"] is True


@pytest.mark.skipif(not _has_key, reason="No LLM API key configured — skipping live integration test")
async def test_real_llm_flags_comparison_intent():
    state = {"question": "Compare vector search and hybrid search for retrieval quality."}
    result = await query_understanding_node(state)

    assert result["attributes"]["requires_comparison"] is True

"""
Unit tests for Module 6. LLM calls mocked.
"""
import sys

import pytest

sys.path.append(".")

from app.core.exceptions import LLMCallError
from app.nodes.generate import (
    NO_CONTEXT_ANSWER,
    _extract_citations,
    _format_context,
    generate_node,
)


# --- empty-docs short-circuit -----------------------------------------------

@pytest.mark.asyncio
async def test_no_documents_skips_llm_and_returns_honest_answer(monkeypatch):
    async def fail_if_called(*a, **k):
        raise AssertionError("LLM should not be called with zero documents")

    monkeypatch.setattr("app.nodes.generate.call_llm", fail_if_called)

    state = {"question": "anything", "reranked_docs": []}
    result = await generate_node(state)

    assert result["answer"] == NO_CONTEXT_ANSWER
    assert result["citations"] == []


# --- normal generation path ---------------------------------------------------

@pytest.mark.asyncio
async def test_generates_answer_and_extracts_citations(monkeypatch):
    async def fake_call_llm(prompt_name, **kwargs):
        assert prompt_name == "final_answer"
        return "LangGraph supports persistence via checkpointers [Source 1]."

    monkeypatch.setattr("app.nodes.generate.call_llm", fake_call_llm)

    state = {
        "question": "Does LangGraph support persistence?",
        "reranked_docs": [{"content": "LangGraph has a checkpointer API.", "source": "docs/persistence.md", "score": 0.9}],
    }
    result = await generate_node(state)

    assert "checkpointers" in result["answer"]
    assert result["citations"] == ["docs/persistence.md"]


@pytest.mark.asyncio
async def test_multiple_citations_deduped_and_ordered_by_first_appearance(monkeypatch):
    async def fake_call_llm(prompt_name, **kwargs):
        return "First point [Source 2]. Second point [Source 1]. Repeat of first [Source 2]."

    monkeypatch.setattr("app.nodes.generate.call_llm", fake_call_llm)

    docs = [
        {"content": "doc A content", "source": "source_A"},
        {"content": "doc B content", "source": "source_B"},
    ]
    state = {"question": "q", "reranked_docs": docs}
    result = await generate_node(state)

    assert result["citations"] == ["source_B", "source_A"]  # order of first appearance, deduped


@pytest.mark.asyncio
async def test_out_of_range_citation_index_ignored_not_crash(monkeypatch):
    async def fake_call_llm(prompt_name, **kwargs):
        return "Some claim [Source 5]."  # only 1 doc exists

    monkeypatch.setattr("app.nodes.generate.call_llm", fake_call_llm)

    state = {
        "question": "q",
        "reranked_docs": [{"content": "doc content", "source": "only_source"}],
    }
    result = await generate_node(state)

    assert result["citations"] == []  # index 5 invalid, silently ignored
    assert "Some claim" in result["answer"]


@pytest.mark.asyncio
async def test_answer_with_no_citation_markers_returns_empty_citations(monkeypatch):
    async def fake_call_llm(prompt_name, **kwargs):
        return "I don't have enough information to answer this confidently."

    monkeypatch.setattr("app.nodes.generate.call_llm", fake_call_llm)

    state = {
        "question": "q",
        "reranked_docs": [{"content": "unrelated content", "source": "doc1"}],
    }
    result = await generate_node(state)
    assert result["citations"] == []


# --- LLM failure: no fallback, propagates --------------------------------------

@pytest.mark.asyncio
async def test_llm_failure_propagates_no_fallback(monkeypatch):
    """Unlike rerank/validation, generation has no sane fallback — must raise."""
    async def failing_call_llm(*a, **k):
        raise LLMCallError("simulated provider outage")

    monkeypatch.setattr("app.nodes.generate.call_llm", failing_call_llm)

    state = {
        "question": "q",
        "reranked_docs": [{"content": "doc content", "source": "doc1"}],
    }
    with pytest.raises(LLMCallError):
        await generate_node(state)


# --- helpers -----------------------------------------------------------------

def test_format_context_numbers_and_truncates():
    docs = [{"content": "x" * 2000, "source": "doc1"}]
    formatted = _format_context(docs)
    assert formatted.startswith("[1]")
    assert len(formatted) < 2700  # truncated below raw context limit (2500 chars + header overhead)


def test_extract_citations_case_insensitive():
    docs = [{"content": "c", "source": "src1"}]
    citations, out_of_range = _extract_citations("claim [source 1] and [SOURCE 1]", docs)
    assert citations == ["src1"]
    assert out_of_range == 0


def test_extract_citations_handles_no_markers_gracefully():
    docs = [{"content": "c", "source": "src1"}]
    citations, out_of_range = _extract_citations("plain answer, no markers", docs)
    assert citations == []
    assert out_of_range == 0

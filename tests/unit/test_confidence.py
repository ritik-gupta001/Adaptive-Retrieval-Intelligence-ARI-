"""
Unit tests for Module 8. No LLM calls exist in this module, so these are
all pure-function tests against deterministic arithmetic — heavier
edge-case coverage is warranted and cheap here.
"""
import sys

import pytest

sys.path.append(".")

from app.nodes.confidence import (
    _blended_confidence,
    _citation_quality,
    _confidence_level,
    _hallucination_risk,
    _retrieval_quality,
    confidence_node,
)


# --- _retrieval_quality -------------------------------------------------------

def test_retrieval_quality_averages_relevance_and_coverage():
    v = {"relevance_score": 0.8, "coverage_score": 0.6}
    assert _retrieval_quality(v) == pytest.approx(0.7)


def test_retrieval_quality_defaults_when_validation_missing():
    assert _retrieval_quality({}) == 0.5


# --- _hallucination_risk -------------------------------------------------------

def test_hallucination_risk_zero_when_clean():
    r = {"is_supported": True, "hallucinations": [], "unsupported_claims": []}
    assert _hallucination_risk(r) == 0.0


def test_hallucination_risk_high_with_hallucinations():
    r = {"is_supported": False, "hallucinations": ["fake fact"], "unsupported_claims": []}
    risk = _hallucination_risk(r)
    assert risk >= 0.5


def test_hallucination_risk_moderate_with_only_unsupported_claims():
    r = {"is_supported": True, "hallucinations": [], "unsupported_claims": ["plausible but unverified"]}
    risk = _hallucination_risk(r)
    assert 0.0 < risk < 0.5


def test_hallucination_risk_increases_with_more_hallucinations():
    r1 = {"is_supported": False, "hallucinations": ["a"], "unsupported_claims": []}
    r2 = {"is_supported": False, "hallucinations": ["a", "b", "c"], "unsupported_claims": []}
    assert _hallucination_risk(r2) > _hallucination_risk(r1)


def test_hallucination_risk_capped_at_one():
    r = {"is_supported": False, "hallucinations": ["a"] * 20, "unsupported_claims": ["b"] * 20}
    assert _hallucination_risk(r) <= 1.0


def test_hallucination_risk_flags_unsupported_overall_with_no_itemized_list():
    """Edge case: reflection says is_supported=False but somehow didn't
    itemize any specific hallucination/unsupported claim — should still
    register meaningful risk, not silently report 0."""
    r = {"is_supported": False, "hallucinations": [], "unsupported_claims": []}
    assert _hallucination_risk(r) > 0.0


def test_hallucination_risk_neutral_when_reflection_missing():
    assert _hallucination_risk({}) == 0.5


# --- _citation_quality ----------------------------------------------------------

def test_citation_quality_zero_with_no_documents():
    assert _citation_quality(["src1"], [], {}) == 0.0


def test_citation_quality_low_but_nonzero_with_docs_but_no_citations_and_supported():
    docs = [{"content": "x"}]
    result = _citation_quality([], docs, {"is_supported": True})
    assert result == 0.2


def test_citation_quality_zero_with_docs_no_citations_and_unsupported():
    """Edge case: worst combination — docs existed, answer cited nothing,
    AND reflection says it's not even supported."""
    docs = [{"content": "x"}]
    result = _citation_quality([], docs, {"is_supported": False})
    assert result == 0.0


def test_citation_quality_high_with_good_citation_ratio():
    docs = [{"content": "x"}, {"content": "y"}]
    result = _citation_quality(["src1", "src2"], docs, {"is_supported": True})
    assert result == 1.0


def test_citation_quality_one_citation_out_of_many_docs_not_punished_too_harshly():
    docs = [{"content": f"doc{i}"} for i in range(10)]
    result = _citation_quality(["src1"], docs, {"is_supported": True})
    assert result > 0.3  # floor protects against near-zero scoring


# --- _blended_confidence / _confidence_level ------------------------------------

def test_blended_confidence_perfect_inputs_near_one():
    score = _blended_confidence(reflection_score=1.0, retrieval_quality=1.0, hallucination_risk=0.0, citation_quality=1.0)
    assert score == pytest.approx(1.0)


def test_blended_confidence_worst_inputs_near_zero():
    score = _blended_confidence(reflection_score=0.0, retrieval_quality=0.0, hallucination_risk=1.0, citation_quality=0.0)
    assert score == pytest.approx(0.0)


def test_blended_confidence_clamped_within_bounds():
    score = _blended_confidence(reflection_score=2.0, retrieval_quality=2.0, hallucination_risk=-1.0, citation_quality=2.0)
    assert 0.0 <= score <= 1.0


def test_hallucination_pulls_score_down_meaningfully():
    """The whole point of weighting hallucination-inverse at 0.25: a single
    bad hallucination should visibly move the needle even with otherwise
    decent scores."""
    clean = _blended_confidence(reflection_score=0.8, retrieval_quality=0.8, hallucination_risk=0.0, citation_quality=0.8)
    hallucinated = _blended_confidence(reflection_score=0.8, retrieval_quality=0.8, hallucination_risk=0.8, citation_quality=0.8)
    assert clean - hallucinated >= 0.15


def test_confidence_level_buckets():
    assert _confidence_level(0.9) == "high"
    assert _confidence_level(0.75) == "high"
    assert _confidence_level(0.6) == "medium"
    assert _confidence_level(0.5) == "medium"
    assert _confidence_level(0.3) == "low"
    assert _confidence_level(0.0) == "low"


# --- confidence_node (full integration of pure functions) ----------------------

@pytest.mark.asyncio
async def test_confidence_node_high_confidence_clean_case():
    state = {
        "question": "What is LangGraph?",
        "validation": {"relevance_score": 0.9, "coverage_score": 0.85},
        "reflection": {
            "is_supported": True,
            "hallucinations": [],
            "unsupported_claims": [],
            "overall_score": 0.9,
        },
        "citations": ["doc1"],
        "reranked_docs": [{"content": "x", "source": "doc1"}],
    }
    result = await confidence_node(state)
    c = result["confidence"]
    assert c["confidence_level"] in ("high", "medium")
    assert c["num_sources"] == 1
    assert c["hallucination_risk"] == 0.0


@pytest.mark.asyncio
async def test_confidence_node_low_confidence_hallucinated_case():
    state = {
        "question": "When was X released?",
        "validation": {"relevance_score": 0.4, "coverage_score": 0.3},
        "reflection": {
            "is_supported": False,
            "hallucinations": ["fabricated date"],
            "unsupported_claims": [],
            "overall_score": 0.2,
        },
        "citations": [],
        "reranked_docs": [{"content": "x", "source": "doc1"}],
    }
    result = await confidence_node(state)
    c = result["confidence"]
    assert c["confidence_level"] == "low"
    assert "hallucination" in c["reason"].lower()


@pytest.mark.asyncio
async def test_confidence_node_handles_completely_empty_state_without_crashing():
    """Edge case: defensive coding for missing upstream fields — shouldn't
    happen given graph order, but the node should never crash on it."""
    state = {"question": "q"}
    result = await confidence_node(state)
    c = result["confidence"]
    assert 0.0 <= c["confidence_score"] <= 1.0
    assert c["num_sources"] == 0


@pytest.mark.asyncio
async def test_confidence_node_no_context_case_scores_low():
    """Mirrors Module 7's no-context short-circuit output."""
    state = {
        "question": "q",
        "validation": {"relevance_score": 0.0, "coverage_score": 0.0, "recommendation": "rewrite"},
        "reflection": {
            "is_supported": True,
            "hallucinations": [],
            "unsupported_claims": [],
            "overall_score": 0.3,
            "should_retry": True,
        },
        "citations": [],
        "reranked_docs": [],
    }
    result = await confidence_node(state)
    assert result["confidence"]["confidence_level"] == "low"
    assert result["confidence"]["citation_quality"] == 0.0

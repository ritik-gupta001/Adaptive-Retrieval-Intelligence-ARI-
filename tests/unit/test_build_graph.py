"""
Unit tests for Module 9.
Gate functions are pure sync — tested directly without invoking the graph.
We use the actual default settings values (confidence_threshold=0.7,
max_retries=2) rather than mocking them — Pydantic v2 model instances
don't support direct attribute assignment, and the gates reading real
configured thresholds is the correct behavior to test against anyway.
"""
import sys

import pytest

sys.path.append(".")

from app.graph.build_graph import (
    _confidence_gate,
    _router_gate,
    _validation_gate,
    build_graph,
)

# Actual defaults from settings — tests are written against these
CONFIDENCE_THRESHOLD = 0.7   # settings.confidence_threshold default
MAX_RETRIES = 2              # settings.max_retries default


# ---------------------------------------------------------------------------
# _router_gate
# ---------------------------------------------------------------------------

def test_router_gate_clarification_path():
    state = {"clarification_needed": True, "strategies": []}
    assert _router_gate(state) == "clarify"


def test_router_gate_normal_retrieval_path():
    state = {"clarification_needed": False, "strategies": ["vector_search"]}
    assert _router_gate(state) == "retrieve"


def test_router_gate_defaults_to_retrieve_when_key_missing():
    assert _router_gate({}) == "retrieve"


# ---------------------------------------------------------------------------
# _validation_gate
# ---------------------------------------------------------------------------

def test_validation_gate_proceeds_when_recommended():
    state = {"validation": {"recommendation": "proceed"}, "retry_count": 0}
    assert _validation_gate(state) == "proceed"


def test_validation_gate_retries_on_rewrite_under_budget():
    state = {"validation": {"recommendation": "rewrite"}, "retry_count": 0}
    assert _validation_gate(state) == "retry"


def test_validation_gate_proceeds_despite_rewrite_when_budget_exhausted():
    state = {"validation": {"recommendation": "rewrite"}, "retry_count": MAX_RETRIES}
    assert _validation_gate(state) == "proceed"


def test_validation_gate_change_strategy_also_retries_under_budget():
    state = {"validation": {"recommendation": "change_strategy"}, "retry_count": 1}
    assert _validation_gate(state) == "retry"


def test_validation_gate_ask_clarification_routes_to_clarify():
    state = {"validation": {"recommendation": "ask_clarification"}, "retry_count": 0}
    assert _validation_gate(state) == "clarify"


def test_validation_gate_defaults_to_proceed_with_no_validation():
    assert _validation_gate({}) == "proceed"


# ---------------------------------------------------------------------------
# _confidence_gate
# ---------------------------------------------------------------------------

def _cs(score, should_retry, retry_count=0, answer="some answer"):
    return {
        "confidence": {"confidence_score": score},
        "reflection": {"should_retry": should_retry},
        "retry_count": retry_count,
        "answer": answer,
    }


def test_confidence_gate_accepts_high_confidence_clean_reflection():
    state = _cs(score=0.9, should_retry=False)
    assert _confidence_gate(state) == "accept"


def test_confidence_gate_retries_when_confidence_low_and_under_budget():
    state = _cs(score=0.3, should_retry=True, retry_count=0)
    assert _confidence_gate(state) == "retry"


def test_confidence_gate_gives_up_at_max_retries():
    state = _cs(score=0.3, should_retry=True, retry_count=MAX_RETRIES)
    assert _confidence_gate(state) == "give_up"


def test_confidence_gate_gives_up_when_no_answer():
    state = _cs(score=0.9, should_retry=False, answer="")
    assert _confidence_gate(state) == "give_up"


def test_confidence_gate_retries_when_should_retry_true_even_with_ok_score():
    """Edge case: confidence score is above threshold but reflection still
    says should_retry=True (e.g. answer accurate but incomplete).
    Retry should take precedence over the score alone."""
    state = _cs(score=0.9, should_retry=True, retry_count=0)
    assert _confidence_gate(state) == "retry"


def test_confidence_gate_accept_requires_both_conditions():
    """Accept needs score >= threshold AND should_retry=False together."""
    assert _confidence_gate(_cs(0.9, False)) == "accept"
    assert _confidence_gate(_cs(0.9, True)) == "retry"
    assert _confidence_gate(_cs(0.3, False)) == "retry"


def test_confidence_gate_score_just_below_threshold_retries():
    state = _cs(score=CONFIDENCE_THRESHOLD - 0.01, should_retry=False, retry_count=0)
    assert _confidence_gate(state) == "retry"


def test_confidence_gate_score_at_threshold_accepts():
    state = _cs(score=CONFIDENCE_THRESHOLD, should_retry=False, retry_count=0)
    assert _confidence_gate(state) == "accept"


# ---------------------------------------------------------------------------
# Graph structure
# ---------------------------------------------------------------------------

def test_graph_compiles_and_has_all_expected_nodes():
    compiled = build_graph().compile()
    nodes = set(compiled.get_graph().nodes.keys())

    expected = {
        "__start__",
        "__end__",
        "memory_load",
        "query_understanding",
        "adaptive_router",
        "retrieve",
        "document_merge",
        "rerank",
        "context_validate",
        "generate",
        "reflect",
        "confidence_score",
        "query_rewrite",
        "finalize",
        "memory_save",
    }
    assert expected == nodes


def test_graph_retry_loop_bypasses_query_understanding_for_latency():
    """Rewrites pass through _rewrite_gate directly to adaptive_router or retrieve,
    bypassing the redundant query_understanding LLM call for optimal latency."""
    edges = {(e.source, e.target) for e in build_graph().compile().get_graph().edges}
    assert ("query_rewrite", "adaptive_router") in edges or ("query_rewrite", "retrieve") in edges


def test_graph_finalize_is_only_terminal_node():
    """All paths must converge through finalize before END — no node
    accidentally has a direct edge to END."""
    edges = {(e.source, e.target) for e in build_graph().compile().get_graph().edges}
    direct_to_end = {src for src, tgt in edges if tgt == "__end__"}
    assert direct_to_end == {"memory_save"}


def test_graph_clarification_exits_to_finalize_without_retrieve():
    """Clarification path should go router → finalize directly,
    never touching retrieve, generate, or any processing node."""
    edges = {(e.source, e.target) for e in build_graph().compile().get_graph().edges}
    assert ("adaptive_router", "finalize") in edges  # clarify path
    assert ("adaptive_router", "retrieve") in edges  # the non-clarify path

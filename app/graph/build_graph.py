from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.config.settings import settings
from app.graph.state import GraphState
from app.nodes.confidence import confidence_node
from app.nodes.context_validate import context_validate_node
from app.nodes.document_merge import document_merge_node
from app.nodes.finalize import finalize_node
from app.nodes.generate import generate_node
from app.nodes.memory_load import memory_load_node
from app.nodes.memory_save import memory_save_node
from app.nodes.query_rewrite import query_rewrite_node
from app.nodes.query_understanding import query_understanding_node
from app.nodes.rerank import rerank_node
from app.nodes.retrieve import retrieve_node
from app.nodes.router import adaptive_router_node
from app.nodes.reflect import reflect_node


def get_recursion_limit() -> int:
    """Dynamically derive the graph recursion limit based on base node count and max retries."""
    base_node_count = 13  # total nodes in the StateGraph
    return base_node_count * (settings.max_retries + 1) + 15


# Gate functions — pure, synchronous, importable and directly testable


def _router_gate(state: GraphState) -> str:
    """After adaptive_router: branch on whether the router asked for
    clarification (no usable retrieval strategy) or is ready to retrieve."""
    if state.get("clarification_needed"):
        return "clarify"
    return "retrieve"


def _validation_gate(state: GraphState) -> str:
    """After context_validate: branch on the validator's recommendation.
    'rewrite' and 'change_strategy' funnel to query_rewrite.

    Cap retries if max_retries reached OR if local_retrieval_failed is already True
    and web search has retrieved docs — proceed to generation instead of looping."""
    recommendation = state.get("validation", {}).get("recommendation", "proceed")
    retry_count = state.get("retry_count", 0)
    docs = state.get("reranked_docs") or state.get("merged_docs") or []

    if recommendation in ("rewrite", "change_strategy"):
        # If local retrieval failed and we already retrieved web search docs, proceed to generate
        if state.get("local_retrieval_failed") and docs:
            return "proceed"
        if retry_count < settings.max_retries:
            return "retry"
        return "proceed"

    if recommendation == "ask_clarification":
        return "clarify"

    return "proceed"


def _confidence_gate(state: GraphState) -> str:
    """After confidence_node: final decision on accept, retry, or give_up."""
    confidence = state.get("confidence", {})
    reflection = state.get("reflection", {})
    retry_count = state.get("retry_count", 0)

    score = confidence.get("confidence_score", 0.0)
    should_retry = reflection.get("should_retry", False)
    is_supported = reflection.get("is_supported", True)

    if not state.get("answer") or retry_count >= settings.max_retries:
        return "give_up"

    # Accept if score is above threshold AND should_retry is False
    if score >= settings.confidence_threshold and not should_retry:
        return "accept"

    # Also accept if local_retrieval_failed (web fallback) and supported
    if state.get("local_retrieval_failed") and is_supported and not should_retry:
        return "accept"

    return "retry"


def _rewrite_gate(state: GraphState) -> str:
    """After query_rewrite: if fallback strategy is set, go directly to 'retrieve'.
    Otherwise go to 'adaptive_router', bypassing the redundant 'query_understanding' LLM call."""
    if state.get("strategies"):
        return "retrieve"
    return "adaptive_router"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    g = StateGraph(GraphState)

    # --- nodes ---
    g.add_node("memory_load", memory_load_node)        # NEW — must be first
    g.add_node("query_understanding", query_understanding_node)
    g.add_node("adaptive_router", adaptive_router_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("document_merge", document_merge_node)
    g.add_node("rerank", rerank_node)
    g.add_node("context_validate", context_validate_node)
    g.add_node("generate", generate_node)
    g.add_node("reflect", reflect_node)
    g.add_node("confidence_score", confidence_node)
    g.add_node("query_rewrite", query_rewrite_node)
    g.add_node("finalize", finalize_node)
    g.add_node("memory_save", memory_save_node)        # NEW — must be last

    # --- linear entry (memory_load now precedes query_understanding) ---
    g.add_edge(START, "memory_load")
    g.add_edge("memory_load", "query_understanding")
    g.add_edge("query_understanding", "adaptive_router")

    # --- router gate: clarify vs retrieve ---
    g.add_conditional_edges(
        "adaptive_router",
        _router_gate,
        {"clarify": "finalize", "retrieve": "retrieve"},
    )

    # --- linear retrieval + processing pipeline ---
    g.add_edge("retrieve", "document_merge")
    g.add_edge("document_merge", "rerank")
    g.add_edge("rerank", "context_validate")

    # --- validation gate: proceed vs retry vs clarify ---
    g.add_conditional_edges(
        "context_validate",
        _validation_gate,
        {"proceed": "generate", "retry": "query_rewrite", "clarify": "finalize"},
    )

    # --- linear generation + evaluation pipeline ---
    g.add_edge("generate", "reflect")
    g.add_edge("reflect", "confidence_score")

    # --- confidence gate: accept vs retry vs give_up ---
    g.add_conditional_edges(
        "confidence_score",
        _confidence_gate,
        {"accept": "finalize", "retry": "query_rewrite", "give_up": "finalize"},
    )

    # --- retry loop: rewrite bypasses redundant query_understanding ---
    g.add_conditional_edges(
        "query_rewrite",
        _rewrite_gate,
        {"retrieve": "retrieve", "adaptive_router": "adaptive_router"},
    )

    # --- terminal (finalize -> memory_save -> END) ---
    g.add_edge("finalize", "memory_save")
    g.add_edge("memory_save", END)

    return g


@lru_cache(maxsize=1)
def get_compiled_graph():
    """Compiled graph singleton. Checkpointer injected here so every
    invocation automatically snapshots GraphState after each node.
    Pass thread_id in RunnableConfig for multi-turn memory:
        graph.invoke(state, config={"configurable": {"thread_id": conv_id}})
    """
    from app.memory.checkpointer import get_checkpointer
    return build_graph().compile(checkpointer=get_checkpointer())

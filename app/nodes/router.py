"""
Node: Adaptive Router.
Determines retrieval strategies using deterministic rule hints combined with LLM routing,
defaulting to local hybrid search over the pre-indexed corpus before web search fallbacks.
"""
import json
from typing import List

from app.config.settings import settings
from app.core.exceptions import LLMCallError
from app.core.logging import get_logger
from app.graph.state import GraphState
from app.core.llm import call_llm_json
from app.utils.query_intent import is_document_reference_query

logger = get_logger(__name__)


def _rule_based_hints(attrs: dict, question: str = "") -> List[str]:
    """Pure, deterministic pre-filter for explicit structural signals.

    Strategy mapping rationale:
    - freshness_needed          → web_search   (must be live data)
    - requires_comparison       → hybrid_search (BM25+vector gives broad coverage)
    - requires_multiple_sources → hybrid_search (NOT multi_query; hybrid already
                                   combines two retrieval axes — multi_query is
                                   reserved for research+complex only)
    - research + complex        → multi_query_retrieval (genuinely multi-faceted)
    - document reference query  → hybrid_search (keyword matching matters here)
    """
    hints: List[str] = []
    intent = attrs.get("intent", "factual")
    complexity = attrs.get("complexity", "simple")

    if attrs.get("freshness_needed"):
        hints.append("web_search")

    if attrs.get("requires_comparison") or intent == "comparison":
        hints.append("hybrid_search")

    # requires_multiple_sources → broader coverage via hybrid, NOT multi_query
    if attrs.get("requires_multiple_sources") and "hybrid_search" not in hints:
        hints.append("hybrid_search")

    # multi_query only for genuinely multi-faceted research at complex depth
    if intent == "research" and complexity == "complex":
        hints.append("multi_query_retrieval")

    if question and is_document_reference_query(question):
        if "hybrid_search" not in hints:
            hints.append("hybrid_search")

    return hints


def _build_clarification_question(question: str) -> str:
    """Deterministic template — no LLM call needed for this. Keeping it
    template-based (not LLM-generated) makes the clarification path fast,
    free, and impossible to hallucinate a confusing follow-up question."""
    trimmed = question.strip()
    if len(trimmed) > 80:
        trimmed = trimmed[:77] + "..."
    return (
        f"Could you clarify what you mean by \"{trimmed}\"? "
        "For example, what specifically would you like to know, or which "
        "aspect are you most interested in?"
    )


async def adaptive_router_node(state: GraphState) -> GraphState:
    attrs = state.get("attributes") or {}
    question = state["question"]
    available = settings.enabled_strategies

    # If state already has web_search fallback set by context_validate, preserve it!
    if state.get("strategies") == ["web_search"]:
        return {**state, "clarification_needed": False, "clarification_question": None}

    rule_hints = [s for s in _rule_based_hints(attrs, question) if s in available]

    # --- Clarification gate ---
    if attrs.get("is_ambiguous") and not rule_hints:
        clarification_q = _build_clarification_question(question)
        logger.info(
            "router_clarification_triggered",
            extra={"question": question, "clarification": clarification_q},
        )
        return {
            **state,
            "clarification_needed": True,
            "clarification_question": clarification_q,
            "strategies": [],
            "issues_log": state.get("issues_log", []) + ["router:clarification_needed"],
        }

    # --- LLM Router Decision ---
    llm_strategies: List[str] = []
    try:
        raw = await call_llm_json(
            "router",
            question=question,
            attributes_json=json.dumps(attrs),
            available_strategies=json.dumps(available),
            max_tokens=settings.max_tokens_classification,
            per_attempt_timeout=settings.llm_classification_timeout,
        )
        for s in raw.get("strategies", []):
            if s in available and s not in llm_strategies:
                llm_strategies.append(s)
    except Exception as exc:
        logger.warning(
            "router_llm_failed_falling_back_to_rules",
            extra={"question": question, "error": str(exc), "rule_hints": rule_hints},
        )

    # Local-First Default: If LLM returned empty and no rule hints, default to local hybrid_search
    combined = llm_strategies + rule_hints
    if not combined:
        default_strat = "hybrid_search" if "hybrid_search" in available else "vector_search"
        combined = [default_strat]

    final_strategies = list(dict.fromkeys(combined))

    logger.info(
        "router_decision",
        extra={
            "question": question,
            "rule_hints": rule_hints,
            "llm_strategies": llm_strategies,
            "final_strategies": final_strategies,
        },
    )

    return {
        **state,
        "clarification_needed": False,
        "clarification_question": None,
        "strategies": final_strategies,
    }

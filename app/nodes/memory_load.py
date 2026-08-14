"""
Node: Memory Load.
Runs BEFORE query_understanding. Reads long-term context from the Store
and injects it into state.long_term_context so every downstream node
(especially the LLM generator and router) has access to:
  - past conversation turns (rolling summary)
  - user's inferred preferences (which strategies worked before)
  - a cached rewrite if this exact question was successfully rewritten before

Failure posture: Store reads failing must NEVER block the pipeline.
Long-term memory is a quality enhancement, not a correctness requirement.
Any Store error is logged and silently ignored — the pipeline proceeds
with an empty long_term_context rather than failing the request.
"""
import uuid
from app.config.settings import settings
from app.core.logging import current_run_id, get_logger
from app.graph.state import GraphState
from app.memory.store import get_store

logger = get_logger(__name__)


async def memory_load_node(state: GraphState) -> GraphState:
    run_id = state.get("run_id") or str(uuid.uuid4())
    current_run_id.set(run_id)

    conversation_id = state.get("conversation_id", "default")
    question = state.get("question", "")
    store = get_store()

    long_term_context: dict = {}

    try:
        summary = await store.get_summary(conversation_id)
        if summary and summary.turns:
            long_term_context["conversation_history"] = [
                {
                    "question": t.question,
                    "answer": t.answer[:settings.memory_answer_preview_chars],  # truncated — context, not verbatim replay
                    "confidence": t.confidence_score,
                }
                for t in summary.turns[-settings.memory_history_turns:]  # configurable history window
            ]
    except Exception as exc:  # noqa: BLE001
        logger.warning("memory_load_summary_failed", extra={"error": str(exc)})

    try:
        prefs = await store.get_preferences(conversation_id)
        if prefs:
            long_term_context["user_preferences"] = {
                "preferred_strategies": prefs.preferred_strategies,
                "preferred_domains": prefs.preferred_domains,
            }
    except Exception as exc:  # noqa: BLE001
        logger.warning("memory_load_preferences_failed", extra={"error": str(exc)})

    try:
        cached_rewrite = await store.get_rewrite(question)
        if cached_rewrite:
            long_term_context["cached_rewrite"] = {
                "successful_rewrite": cached_rewrite.successful_rewrite,
                "strategies_that_worked": cached_rewrite.strategies_that_worked,
                "success_count": cached_rewrite.success_count,
            }
            logger.info(
                "memory_load_rewrite_cache_hit",
                extra={
                    "question": question,
                    "cached_rewrite": cached_rewrite.successful_rewrite,
                    "success_count": cached_rewrite.success_count,
                },
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("memory_load_rewrite_failed", extra={"error": str(exc)})

    logger.info(
        "memory_loaded",
        extra={
            "conversation_id": conversation_id,
            "has_history": "conversation_history" in long_term_context,
            "has_preferences": "user_preferences" in long_term_context,
            "has_cached_rewrite": "cached_rewrite" in long_term_context,
        },
    )

    return {**state, "long_term_context": long_term_context}

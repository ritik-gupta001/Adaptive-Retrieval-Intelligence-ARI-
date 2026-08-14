"""
Node: Query Rewrite.
Rewrites the user query when retrieval validation or reflection fails, clearing downstream state for retry.

Fallback: if the LLM rewrite call itself fails, we fall back to the
original question rather than raising — the rewrite is a quality
improvement, not a correctness requirement for the retry itself.
"""
from app.config.settings import settings
from app.core.exceptions import LLMCallError
from app.core.llm import call_llm
from app.core.logging import get_logger
from app.graph.state import GraphState

logger = get_logger(__name__)

_STALE_FIELDS = [
    "retrieved_docs",
    "merged_docs",
    "reranked_docs",
    "validation",
    "answer",
    "citations",
    "reflection",
    "confidence",
]


async def query_rewrite_node(state: GraphState) -> GraphState:
    retry_count = state.get("retry_count", 0) + 1
    original = state.get("original_question", state["question"])
    current = state["question"]
    issues = state.get("issues_log", [])
    issues_str = "; ".join(issues[-5:]) if issues else "no specific issues identified"

    from app.memory.store import get_store
    store = get_store()

    cached_entry = await store.get_rewrite(current)
    if cached_entry and cached_entry.successful_rewrite:
        rewritten = cached_entry.successful_rewrite
        logger.info(
            "query_rewrite_cache_hit",
            extra={"original": current, "rewritten": rewritten},
        )
    else:
        try:
            rewritten = await call_llm(
                "query_rewrite", question=current, issues=issues_str
            )
            rewritten = rewritten.strip().strip('"').strip("'")
            if not rewritten:
                raise LLMCallError("empty rewrite returned")
            await store.save_rewrite(original, rewritten, state.get("strategies", []))
        except LLMCallError as exc:
            logger.warning(
                "query_rewrite_failed_using_original",
                extra={"question": current, "retry_count": retry_count, "error": str(exc)},
            )
            rewritten = original  # fall back to original question

    logger.info(
        "query_rewritten",
        extra={
            "original": original,
            "rewritten": rewritten,
            "retry_count": retry_count,
            "max_retries": settings.max_retries,
        },
    )

    # Clear stale downstream state so the new pass starts completely fresh.
    #
    # STRATEGY PRESERVATION LOGIC:
    # "strategies" is intentionally NOT in _STALE_FIELDS.
    # context_validate_node may have set strategies=["web_search"] as a
    # deliberate fallback when local retrieval was weak/irrelevant.
    # If we cleared strategies here, the retry would re-enter adaptive_router
    # which would pick local strategies again — defeating the entire
    # local-failure recovery path.
    #
    # Rule:
    #   - local_retrieval_failed=True  → keep current strategies (web_search fallback)
    #   - local_retrieval_failed=False → clear strategies so router re-decides freshly
    cleared = {field: None for field in _STALE_FIELDS}

    if state.get("local_retrieval_failed"):
        # Preserve the fallback strategy the context validator selected
        cleared["strategies"] = state.get("strategies")
        logger.info(
            "query_rewrite_preserving_fallback_strategy",
            extra={"strategies": cleared["strategies"], "retry_count": retry_count},
        )
    else:
        # Let the router re-decide with the rewritten query
        cleared["strategies"] = None

    return {
        **state,
        **cleared,
        "question": rewritten,
        "original_question": original,
        "retry_count": retry_count,
        "issues_log": state.get("issues_log", [])
        + [f"retry_{retry_count}:rewritten_query='{rewritten}'"],
    }

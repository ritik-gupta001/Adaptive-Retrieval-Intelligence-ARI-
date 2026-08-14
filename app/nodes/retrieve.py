"""
Node: Retrieve.
Fans out to every strategy in state.strategies concurrently. A failure in
one strategy does not fail the node — only logged and excluded — UNLESS
every selected strategy fails, in which case there's genuinely nothing to
retrieve and we raise (no sane fallback for "zero documents found").
"""
import asyncio

from app.core.exceptions import RetrievalError
from app.core.logging import get_logger
from app.config.settings import settings
from app.graph.state import GraphState
from app.retrievers.registry import get_retriever

logger = get_logger(__name__)


async def retrieve_node(state: GraphState) -> GraphState:
    strategies = state.get("strategies", [])
    question = state["question"]

    if not strategies:
        raise RetrievalError(
            "retrieve_node called with no strategies in state — router "
            "should have either set strategies or clarification_needed."
        )

    results = await asyncio.gather(
        *[get_retriever(s).retrieve(question, settings.top_k) for s in strategies],
        return_exceptions=True,
    )

    all_docs = []
    failed_strategies = []
    for strategy, result in zip(strategies, results):
        if isinstance(result, Exception):
            failed_strategies.append(strategy)
            logger.warning(
                "retrieval_strategy_failed",
                extra={"strategy": strategy, "question": question, "error": str(result)},
            )
            continue
        all_docs.extend(result)

    if len(failed_strategies) == len(strategies):
        raise RetrievalError(
            f"All selected retrieval strategies failed: {strategies}"
        )

    logger.info(
        "retrieval_completed",
        extra={
            "question": question,
            "strategies_attempted": strategies,
            "strategies_failed": failed_strategies,
            "docs_retrieved": len(all_docs),
        },
    )

    return {
        **state,
        "retrieved_docs": all_docs,
        "issues_log": state.get("issues_log", [])
        + ([f"strategies_failed:{failed_strategies}"] if failed_strategies else []),
    }

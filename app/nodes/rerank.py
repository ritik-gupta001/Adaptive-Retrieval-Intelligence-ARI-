"""
Node: Reranker.
Reranks merged document results using the configured reranker model, falling back to merge order on failure.
"""

from typing import Dict

from app.config.settings import settings
from app.core.exceptions import RerankError
from app.core.logging import get_logger
from app.graph.state import GraphState
from app.rerankers.base import BaseReranker
from app.rerankers.bge_reranker import BGEReranker
from app.rerankers.cohere_reranker import CohereReranker

logger = get_logger(__name__)

_RERANKERS: Dict[str, BaseReranker] = {}


def _get_reranker() -> BaseReranker:
    provider = settings.reranker_provider
    if provider not in _RERANKERS:
        if provider == "bge":
            _RERANKERS[provider] = BGEReranker()
        elif provider == "cohere":
            _RERANKERS[provider] = CohereReranker()
        else:
            raise RerankError(f"Unknown reranker_provider: {provider}")
    return _RERANKERS[provider]


async def rerank_node(state: GraphState) -> GraphState:
    docs = state.get("merged_docs", [])
    top_n = settings.top_k

    if not docs:
        logger.info("rerank_skipped_no_docs")
        return {**state, "reranked_docs": []}

    if settings.reranker_provider == "none":
        logger.info("rerank_skipped_provider_none")
        return {**state, "reranked_docs": docs[:top_n]}

    try:
        reranker = _get_reranker()
        reranked = await reranker.rerank(state["question"], docs, top_n)
        logger.info(
            "rerank_completed",
            extra={"provider": settings.reranker_provider, "docs_in": len(docs), "docs_out": len(reranked)},
        )
        return {**state, "reranked_docs": reranked}
    except RerankError as exc:
        logger.warning(
            "rerank_failed_falling_back_to_merge_order",
            extra={"provider": settings.reranker_provider, "error": str(exc)},
        )
        fallback = docs[:top_n]
        return {
            **state,
            "reranked_docs": fallback,
            "issues_log": state.get("issues_log", []) + [f"rerank_fallback:{exc}"],
        }
    except Exception as exc:
        logger.error(
            "rerank_unexpected_failure",
            extra={"provider": settings.reranker_provider, "error": str(exc)},
        )
        fallback = docs[:top_n]
        return {
            **state,
            "reranked_docs": fallback,
            "issues_log": state.get("issues_log", []) + [f"rerank_unexpected_failure_total:{exc}"],
        }

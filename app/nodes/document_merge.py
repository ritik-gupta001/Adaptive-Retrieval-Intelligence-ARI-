"""
Node: Document Merge.
Combines results from every (successful) strategy in retrieve_node and
deduplicates across them — this is exactly where, e.g., the same chunk
returned by both vector_search and the vector leg of hybrid_search would
otherwise be sent to the reranker twice.
"""
from app.core.logging import get_logger
from app.graph.state import GraphState
from app.utils.dedup import dedupe_documents

logger = get_logger(__name__)


async def document_merge_node(state: GraphState) -> GraphState:
    retrieved = state.get("retrieved_docs", [])
    merged = dedupe_documents(retrieved)

    logger.info(
        "document_merge_completed",
        extra={
            "docs_before_merge": len(retrieved),
            "docs_after_merge": len(merged),
            "strategies_represented": sorted(
                {d.get("strategy", "unknown") for d in merged}
            ),
        },
    )

    return {**state, "merged_docs": merged}

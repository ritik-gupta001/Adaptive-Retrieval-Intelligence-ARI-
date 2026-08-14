"""
Node: Finalize.
Terminal node in the pipeline that formats final output, citations, and confidence state.
"""

from app.core.logging import get_logger
from app.graph.state import GraphState

logger = get_logger(__name__)

_NO_ANSWER = (
    "I wasn't able to find enough reliable information to answer this "
    "confidently after multiple retrieval attempts."
)

_LOW_CONFIDENCE_PREFIX = (
    "[Low confidence — treat with caution]\n\n"
)


async def finalize_node(state: GraphState) -> GraphState:
    retry_count = state.get("retry_count", 0)
    clarification_needed = state.get("clarification_needed", False)
    confidence = state.get("confidence", {})
    answer = state.get("answer", "")

    if clarification_needed:
        exit_path = "clarification"
        final_answer = ""
        final_citations: list = []
    elif not answer:
        exit_path = "give_up_no_answer"
        final_answer = _NO_ANSWER
        final_citations = []
    elif confidence.get("confidence_level") == "low" and retry_count > 0:
        exit_path = "give_up_low_confidence"
        final_answer = _LOW_CONFIDENCE_PREFIX + answer
        final_citations = state.get("citations", [])
    else:
        exit_path = "accept"
        final_answer = answer
        final_citations = state.get("citations", [])

    logger.info(
        "pipeline_finalized",
        extra={
            "exit_path": exit_path,
            "retry_count": retry_count,
            "confidence_score": confidence.get("confidence_score"),
            "num_citations": len(final_citations),
        },
    )

    return {
        **state,
        "answer": final_answer,
        "citations": final_citations,
        "final": True,
    }

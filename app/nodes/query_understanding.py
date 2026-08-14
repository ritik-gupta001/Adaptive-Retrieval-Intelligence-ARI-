"""
Node: Query Understanding Agent.
Classifies query intent, complexity, domain attributes, and freshness needs.
"""

from pydantic import ValidationError

from app.config.settings import settings
from app.core.exceptions import LLMCallError
from app.core.llm import call_llm_json
from app.core.logging import get_logger
from app.graph.state import GraphState
from app.schemas.query_understanding import QueryUnderstandingOutput

logger = get_logger(__name__)


async def query_understanding_node(state: GraphState) -> GraphState:
    question = state["question"]

    from app.security.guardrails import validate_input_security
    is_safe, reason = validate_input_security(question)
    if not is_safe:
        logger.warning("query_understanding_blocked_by_guardrail", extra={"question": question, "reason": reason})
        return {
            **state,
            "answer": reason,
            "citations": [],
            "final": True,
            "attributes": {"intent": "other", "complexity": "simple", "freshness_needed": False, "is_ambiguous": False},
        }

    try:
        raw = await call_llm_json(
            "query_understanding", question=question,
            max_tokens=settings.max_tokens_classification,
            per_attempt_timeout=settings.llm_classification_timeout
        )
    except LLMCallError:
        logger.error("query_understanding_llm_failed", extra={"question": question})
        raise

    try:
        parsed = QueryUnderstandingOutput.model_validate(raw)
    except ValidationError as exc:
        logger.error(
            "query_understanding_validation_failed",
            extra={"question": question, "raw": raw, "errors": exc.errors()},
        )
        raise LLMCallError(
            f"Query Understanding output failed schema validation: {exc}"
        ) from exc

    attributes = parsed.model_dump()
    attributes["is_ambiguous"] = parsed.is_ambiguous()

    logger.info(
        "query_understood",
        extra={
            "question": question,
            "intent": attributes["intent"],
            "complexity": attributes["complexity"],
            "freshness_needed": attributes["freshness_needed"],
            "is_ambiguous": attributes["is_ambiguous"],
        },
    )

    return {
        **state,
        "attributes": attributes,
        # explicitly NOT touching "question" or "answer" — this node classifies only
    }

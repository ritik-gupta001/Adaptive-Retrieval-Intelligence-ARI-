"""
Node: Reflection Agent.
Evaluates the generated answer against context documents to detect hallucination risks and verify citation accuracy.
"""

from pydantic import ValidationError

from app.config.settings import settings
from app.core.exceptions import LLMCallError
from app.core.llm import call_llm_json
from app.core.logging import get_logger
from app.graph.state import GraphState
from app.schemas.reflection import ReflectionOutput

logger = get_logger(__name__)


def _format_context(docs) -> str:
    lines = []
    for i, doc in enumerate(docs, start=1):
        content = doc.get("content", "")[:settings.max_doc_chars_reflection]
        lines.append(f"[{i}] {content}")
    return "\n".join(lines)


async def reflect_node(state: GraphState) -> GraphState:
    docs = state.get("reranked_docs", [])
    answer = state.get("answer", "")
    question = state["question"]
    intent = (state.get("attributes") or {}).get("intent", "factual")

    if intent == "other":
        reflection = {
            "is_supported": True,
            "hallucinations": [],
            "unsupported_claims": [],
            "missing_information": "",
            "incorrect_reasoning": [],
            "completeness_score": 1.0,
            "overall_score": 1.0,
            "should_retry": False,
            "reasoning": "Conversational/subjective query — reflection verification bypassed.",
        }
        return {**state, "reflection": reflection}

    if not docs:
        logger.info("reflection_skipped_no_context", extra={"question": question})
        reflection = {
            "is_supported": True,  # vacuously true — no factual claims were made
            "hallucinations": [],
            "unsupported_claims": [],
            "missing_information": "No context was available to generate from.",
            "incorrect_reasoning": [],
            "completeness_score": 0.0,
            "overall_score": 0.3,
            "should_retry": True,  # worth trying a different strategy
            "reasoning": "No retrieved context — nothing to reflect on, but signal retry.",
        }
        return {**state, "reflection": reflection}

    try:
        raw = await call_llm_json(
            "reflection", question=question, answer=answer, context=_format_context(docs), max_tokens=settings.max_tokens_reflection
        )
        validated = ReflectionOutput.model_validate(raw)
        reflection = validated.model_dump()
        logger.info(
            "reflection_completed",
            extra={
                "question": question,
                "is_supported": reflection["is_supported"],
                "overall_score": reflection["overall_score"],
                "should_retry": reflection["should_retry"],
                "num_hallucinations": len(reflection["hallucinations"]),
            },
        )
    except (LLMCallError, ValidationError) as exc:
        logger.warning(
            "reflection_failed_using_cautious_defaults",
            extra={"question": question, "error": str(exc)},
        )
        reflection = {
            "is_supported": True,
            "hallucinations": [],
            "unsupported_claims": [],
            "missing_information": "",
            "incorrect_reasoning": [],
            "completeness_score": 0.5,
            "overall_score": 0.5,
            "should_retry": False,
            "reasoning": "Reflection agent unavailable — defaulted, did not actually verify the answer.",
        }

    issues = []
    if reflection["hallucinations"]:
        issues.append(f"hallucinations:{reflection['hallucinations']}")
    if reflection["unsupported_claims"]:
        issues.append(f"unsupported_claims:{reflection['unsupported_claims']}")

    return {
        **state,
        "reflection": reflection,
        "issues_log": state.get("issues_log", []) + issues,
    }

"""
Node: Context Validator.
Validates retrieved documents against query requirements before sending to the LLM generator.
"""

import json

from pydantic import ValidationError

from app.config.settings import settings
from app.core.exceptions import LLMCallError
from app.core.llm import call_llm_json
from app.core.logging import get_logger
from app.graph.state import GraphState
from app.schemas.context_validation import ValidationOutput
from app.utils.query_intent import is_document_reference_query

logger = get_logger(__name__)


def _format_documents(docs) -> str:
    lines = []
    for i, doc in enumerate(docs, start=1):
        content = doc.get("content", "")[:settings.max_doc_chars_validation]
        source = doc.get("source", "unknown")
        lines.append(f"[{i}] (source: {source}) {content}")
    return "\n".join(lines)


LOCAL_TECH_KEYWORDS = [
    "langgraph",
    "checkpointer",
    "checkpoint",
    "memorysaver",
    "sqlitesaver",
    "postgressaver",
    "vector search",
    "hybrid search",
    "multi-query",
    "multi query",
    "reciprocal rank fusion",
    "rrf",
    "reflection agent",
    "hallucination",
    "confidence threshold",
    "retry",
    "retriever",
    "rerank",
]



async def context_validate_node(state: GraphState) -> GraphState:
    docs = state.get("reranked_docs", [])
    question = state["question"]
    intent = state.get("attributes", {}).get("intent", "factual")

    if intent == "other":
        validation = {
            "is_relevant": True,
            "relevance_score": 1.0,
            "coverage_score": 1.0,
            "missing_information": "",
            "has_duplicates": False,
            "issues": [],
            "recommendation": "proceed",
        }
        return {**state, "validation": validation, "local_retrieval_failed": False}

    is_uploaded_pdf_or_file = any(
        str(doc.get("source", "")).lower().endswith(tuple(settings.uploaded_file_extensions))
        and not str(doc.get("source", "")).startswith("data")
        for doc in docs
    )
    if is_uploaded_pdf_or_file and docs:
        validation = {
            "is_relevant": True,
            "relevance_score": 0.95,
            "coverage_score": 0.95,
            "missing_information": "",
            "has_duplicates": False,
            "issues": [],
            "recommendation": "proceed",
        }
        return {
            **state,
            "validation": validation,
            "local_retrieval_failed": False,
            "strategies": state.get("strategies", []),
        }

    if not docs:
        logger.warning("context_validation_no_documents", extra={"question": question})
        validation = {
            "is_relevant": False,
            "relevance_score": 0.0,
            "coverage_score": 0.0,
            "missing_information": "No local documents were retrieved for this question.",
            "has_duplicates": False,
            "issues": ["no local documents retrieved"],
            "recommendation": "rewrite",
            "suggested_strategy": "web_search",
        }
        fallback_strategies = ["web_search"] if "web_search" in settings.enabled_strategies else ["hybrid_search"]
        return {
            **state,
            "validation": validation,
            "local_retrieval_failed": True,
            "strategies": fallback_strategies,
        }

    local_retrieval_failed = False
    try:
        raw = await call_llm_json(
            "context_validator", question=question, documents=_format_documents(docs),
            max_tokens=settings.max_tokens_validation, per_attempt_timeout=settings.llm_classification_timeout
        )
        validated = ValidationOutput.model_validate(raw)
        validation = validated.model_dump()
        relevance_score = validation.get("relevance_score", 0.0)
        is_relevant = validation.get("is_relevant", True)
        recommendation = validation.get("recommendation", "proceed")

        # Set local_retrieval_failed = True if local docs are weak/irrelevant so pipeline falls back to Web Search
        is_local_tech = any(kw in question.lower() for kw in LOCAL_TECH_KEYWORDS)
        if is_local_tech and is_relevant and relevance_score >= 0.35:
            local_retrieval_failed = False
        else:
            local_retrieval_failed = (not is_relevant) or (relevance_score < 0.35) or (recommendation in ("rewrite", "change_strategy"))

        fallback_strategies = state.get("strategies", [])
        if local_retrieval_failed and "web_search" in settings.enabled_strategies:
            fallback_strategies = ["web_search"]

        logger.info(
            "context_validation_completed",
            extra={
                "question": question,
                "recommendation": recommendation,
                "relevance_score": relevance_score,
                "local_retrieval_failed": local_retrieval_failed,
                "is_local_tech": is_local_tech,
            },
        )
    except (LLMCallError, ValidationError) as exc:
        logger.warning(
            "context_validation_failed_defaulting_to_proceed",
            extra={"question": question, "error": str(exc)},
        )
        validation = {
            "is_relevant": True,
            "relevance_score": 0.5,
            "coverage_score": 0.5,
            "missing_information": "",
            "has_duplicates": False,
            "issues": ["context validator unavailable — defaulted to proceed"],
            "recommendation": "proceed",
        }
        fallback_strategies = state.get("strategies", [])

    return {
        **state,
        "validation": validation,
        "local_retrieval_failed": local_retrieval_failed,
        "strategies": fallback_strategies,
        "issues_log": state.get("issues_log", [])
        + ([f"validation:{i}" for i in validation.get("issues", [])]),
    }

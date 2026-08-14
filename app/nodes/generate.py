"""
Node: LLM Generator.
Generates long-form answers using retrieved context documents and extracts citations.
"""
import re
from typing import List

from app.config.settings import settings
from app.core.exceptions import LLMCallError
from app.core.llm import call_llm
from app.core.logging import get_logger
from app.graph.state import GraphState

logger = get_logger(__name__)

CITATION_PATTERN = re.compile(r"\[(?:Source\s+)?(\d+)\]", re.IGNORECASE)

NO_CONTEXT_ANSWER = (
    "I don't have enough verified information retrieved to answer this "
    "question confidently. Could you rephrase it, or provide more context?"
)


def _format_context(docs: List[dict]) -> str:
    lines = []
    for i, doc in enumerate(docs, start=1):
        content = doc.get("content", "")[:settings.max_doc_chars_generation]
        source = doc.get("source", "unknown")
        lines.append(f"[{i}] (source: {source})\n{content}")
    return "\n\n".join(lines)


def _extract_citations(answer_text: str, docs: List[dict]) -> tuple:
    """Returns (citations: List[str], out_of_range_count: int).

    out_of_range_count is the number of [N] references in the answer that
    point to an index outside the available doc list. These are surfaced back
    to the caller so the confidence scorer can penalise grounding failures
    rather than silently dropping them.
    """
    seen = set()
    citations: List[str] = []
    out_of_range_count = 0
    for match in CITATION_PATTERN.finditer(answer_text):
        idx = int(match.group(1))
        if 1 <= idx <= len(docs):
            source = docs[idx - 1].get("source", "unknown")
            if source not in seen:
                seen.add(source)
                citations.append(source)
        else:
            out_of_range_count += 1
            logger.warning(
                "citation_index_out_of_range",
                extra={"cited_index": idx, "num_docs": len(docs)},
            )
    return citations, out_of_range_count


async def generate_node(state: GraphState) -> GraphState:
    question = state.get("question", "")
    docs = state.get("reranked_docs") or state.get("merged_docs") or []

    if not docs:
        logger.warning("generation_skipped_no_documents", extra={"question": question})
        return {**state, "answer": NO_CONTEXT_ANSWER, "citations": []}

    memory_context = state.get("long_term_context", {})
    history_lines = []
    if memory_context and isinstance(memory_context, dict) and "conversation_history" in memory_context:
        for turn in memory_context["conversation_history"]:
            history_lines.append(f"User: {turn.get('question', '')}\nAI: {turn.get('answer', '')}")
    memory_str = "\n".join(history_lines) if history_lines else "None"

    try:
        answer_text = await call_llm(
            "final_answer",
            question=question,
            context=_format_context(docs),
            memory_context=memory_str,
            max_tokens=settings.max_tokens_generation,
        )
    except LLMCallError:
        logger.error("generation_failed", extra={"question": question})
        raise

    citations, out_of_range_citations = _extract_citations(answer_text, docs)

    from app.security.guardrails import sanitize_output_security
    sanitized_answer = sanitize_output_security(answer_text)

    logger.info(
        "generation_completed",
        extra={
            "question": question,
            "answer_length": len(sanitized_answer),
            "num_citations": len(citations),
            "out_of_range_citations": out_of_range_citations,
        },
    )

    issues = state.get("issues_log", [])
    if out_of_range_citations:
        issues = issues + [f"citation_out_of_range:{out_of_range_citations}"]

    return {**state, "answer": sanitized_answer, "citations": citations, "issues_log": issues}

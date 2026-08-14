
import asyncio
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from app.config.settings import settings
from app.graph.build_graph import get_compiled_graph
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    """Everything the eval scorers need from one completed pipeline run."""
    question: str
    answer: str
    citations: List[str]
    retrieved_contexts: List[str]   # content strings, not Document dicts
    strategies_used: List[str]
    confidence_score: float
    confidence_level: str
    hallucination_risk: float
    retry_count: int
    reflection_is_supported: bool
    reflection_hallucinations: List[str]
    clarification_needed: bool
    error: Optional[str] = None
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))


async def run_single(question: str) -> PipelineResult:
    """Run one question through the compiled graph and extract eval fields."""
    graph = get_compiled_graph()
    run_id = str(uuid.uuid4())

    initial_state = {
        "question": question,
        "original_question": question,
        "conversation_id": f"eval-{run_id}",
        "run_id": run_id,
        "retry_count": 0,
        "issues_log": [],
    }
    config = {
        "configurable": {"thread_id": f"eval-thread-{run_id}"},
        "recursion_limit": settings.graph_recursion_limit,
    }

    try:
        final_state = await graph.ainvoke(initial_state, config)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "eval_pipeline_run_failed",
            extra={"question": question, "error": str(exc)},
        )
        return PipelineResult(
            question=question,
            answer="",
            citations=[],
            retrieved_contexts=[],
            strategies_used=[],
            confidence_score=0.0,
            confidence_level="low",
            hallucination_risk=1.0,
            retry_count=0,
            reflection_is_supported=False,
            reflection_hallucinations=[],
            clarification_needed=False,
            error=str(exc),
            run_id=run_id,
        )

    reranked = final_state.get("reranked_docs") or []
    retrieved_contexts = [d.get("content", "") for d in reranked]

    confidence = final_state.get("confidence") or {}
    reflection = final_state.get("reflection") or {}

    return PipelineResult(
        question=question,
        answer=final_state.get("answer", ""),
        citations=final_state.get("citations", []),
        retrieved_contexts=retrieved_contexts,
        strategies_used=final_state.get("strategies") or [],
        confidence_score=confidence.get("confidence_score", 0.0),
        confidence_level=confidence.get("confidence_level", "low"),
        hallucination_risk=confidence.get("hallucination_risk", 0.0),
        retry_count=final_state.get("retry_count", 0),
        reflection_is_supported=reflection.get("is_supported", True),
        reflection_hallucinations=reflection.get("hallucinations", []),
        clarification_needed=final_state.get("clarification_needed", False),
        run_id=run_id,
    )


async def run_pipeline(questions: List[str]) -> List[PipelineResult]:
    """Run all questions sequentially. Sequential to avoid LLM rate limits."""
    results = []
    for i, q in enumerate(questions, start=1):
        logger.info(
            "eval_running_query",
            extra={"index": i, "total": len(questions), "question": q[:80]},
        )
        result = await run_single(q)
        results.append(result)
    return results

"""
Run metrics extractor for the observability stack.
Extracts structured performance metrics from completed GraphState dictionaries.
"""

from typing import Any, Dict, List, Optional


class RunMetrics:
    """
    Structured metrics extracted from a completed GraphState.
    Typed as a class (not TypedDict) so we get dot-access in tests and
    the FastAPI response layer, plus a clean .to_dict() for serialization.
    """

    def __init__(
        self,
        *,
        run_id: str,
        question: str,
        strategies_used: List[str],
        strategy_reasoning: str,
        retry_count: int,
        docs_retrieved: int,
        docs_after_merge: int,
        docs_after_rerank: int,
        validation_recommendation: str,
        validation_relevance_score: float,
        reflection_is_supported: bool,
        reflection_hallucinations: List[str],
        reflection_overall_score: float,
        confidence_score: float,
        confidence_level: str,
        hallucination_risk: float,
        retrieval_quality: float,
        citation_quality: float,
        num_citations: int,
        issues_log: List[str],
        clarification_needed: bool,
        had_memory_context: bool,
    ):
        self.run_id = run_id
        self.question = question
        self.strategies_used = strategies_used
        self.strategy_reasoning = strategy_reasoning
        self.retry_count = retry_count
        self.docs_retrieved = docs_retrieved
        self.docs_after_merge = docs_after_merge
        self.docs_after_rerank = docs_after_rerank
        self.validation_recommendation = validation_recommendation
        self.validation_relevance_score = validation_relevance_score
        self.reflection_is_supported = reflection_is_supported
        self.reflection_hallucinations = reflection_hallucinations
        self.reflection_overall_score = reflection_overall_score
        self.confidence_score = confidence_score
        self.confidence_level = confidence_level
        self.hallucination_risk = hallucination_risk
        self.retrieval_quality = retrieval_quality
        self.citation_quality = citation_quality
        self.num_citations = num_citations
        self.issues_log = issues_log
        self.clarification_needed = clarification_needed
        self.had_memory_context = had_memory_context

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "question": self.question,
            "strategies_used": self.strategies_used,
            "strategy_reasoning": self.strategy_reasoning,
            "retry_count": self.retry_count,
            "docs_retrieved": self.docs_retrieved,
            "docs_after_merge": self.docs_after_merge,
            "docs_after_rerank": self.docs_after_rerank,
            "validation_recommendation": self.validation_recommendation,
            "validation_relevance_score": self.validation_relevance_score,
            "reflection_is_supported": self.reflection_is_supported,
            "reflection_hallucinations": self.reflection_hallucinations,
            "reflection_overall_score": self.reflection_overall_score,
            "confidence_score": self.confidence_score,
            "confidence_level": self.confidence_level,
            "hallucination_risk": self.hallucination_risk,
            "retrieval_quality": self.retrieval_quality,
            "citation_quality": self.citation_quality,
            "num_citations": self.num_citations,
            "issues_log": self.issues_log,
            "clarification_needed": self.clarification_needed,
            "had_memory_context": self.had_memory_context,
        }

    @property
    def is_high_quality(self) -> bool:
        """Quick boolean gate used by memory_save to decide whether to
        persist this turn — same logic as Module 10's write threshold,
        here as a convenience property for the API layer too."""
        return (
            self.confidence_level in ("medium", "high")
            and self.reflection_is_supported
            and not self.reflection_hallucinations
        )


def extract_run_metrics(state: Dict[str, Any]) -> RunMetrics:
    """
    Pure extraction — reads every relevant field from the completed
    GraphState and assembles a RunMetrics. Defaults every field so this
    never raises even if a node upstream short-circuited and left
    some fields unpopulated (e.g. clarification path skips retrieval).
    """
    confidence = state.get("confidence") or {}
    reflection = state.get("reflection") or {}
    validation = state.get("validation") or {}
    long_term_context = state.get("long_term_context") or {}

    retrieved = state.get("retrieved_docs") or []
    merged = state.get("merged_docs") or []
    reranked = state.get("reranked_docs") or []
    citations = state.get("citations") or []
    issues = state.get("issues_log") or []

    return RunMetrics(
        run_id=state.get("run_id", "unknown"),
        question=state.get("original_question") or state.get("question", ""),
        strategies_used=state.get("strategies") or [],
        strategy_reasoning=state.get("strategy_reasoning", ""),
        retry_count=state.get("retry_count", 0),
        docs_retrieved=len(retrieved),
        docs_after_merge=len(merged),
        docs_after_rerank=len(reranked),
        validation_recommendation=validation.get("recommendation", "unknown"),
        validation_relevance_score=validation.get("relevance_score", 0.0),
        reflection_is_supported=reflection.get("is_supported", True),
        reflection_hallucinations=reflection.get("hallucinations", []),
        reflection_overall_score=reflection.get("overall_score", 0.0),
        confidence_score=confidence.get("confidence_score", 0.0),
        confidence_level=confidence.get("confidence_level", "low"),
        hallucination_risk=confidence.get("hallucination_risk", 0.0),
        retrieval_quality=confidence.get("retrieval_quality", 0.0),
        citation_quality=confidence.get("citation_quality", 0.0),
        num_citations=len(citations),
        issues_log=issues,
        clarification_needed=state.get("clarification_needed", False),
        had_memory_context=bool(long_term_context),
    )

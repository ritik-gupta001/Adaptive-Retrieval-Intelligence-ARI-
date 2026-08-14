"""
Node: Confidence Estimation.
Computes evidence-based, query-dependent confidence scores based on document relevance scores,
retrieval validation, answer reflection, citation coverage, and query attributes.
"""

from typing import List
from app.config.settings import settings
from app.core.logging import get_logger
from app.graph.state import GraphState

logger = get_logger(__name__)


import math

def _normalize_doc_score(doc: dict) -> float:
    s = doc.get("score")
    if s is None or not isinstance(s, (int, float)):
        return 0.5
    s = float(s)
    strategy = str(doc.get("strategy", "")).lower()

    # Reranker (BGE cross-encoder) logits or unbounded scores: apply sigmoid
    if "rerank" in strategy or s > 1.0 or s < 0.0:
        try:
            return 1.0 / (1.0 + math.exp(-s))
        except OverflowError:
            return 1.0 if s > 0 else 0.0

    # RRF fusion scores (small fractions): normalize against theoretical max for k=60 with 2 lists
    if "hybrid" in strategy or "rrf" in strategy:
        max_rrf = 2.0 / 61.0
        return min(1.0, max(0.0, s / max_rrf))

    # Vector (cosine) and BM25 scores are already in [0, 1]
    return max(0.0, min(1.0, s))


def _doc_evidence_score(docs: List[dict]) -> float:
    if not docs:
        return 0.2
    scores = [_normalize_doc_score(d) for d in docs]
    if not scores:
        return 0.5
    top_score = max(scores)
    avg_score = sum(scores) / len(scores)

    evidence = 0.6 * top_score + 0.4 * avg_score
    return max(0.1, min(1.0, evidence))


def _retrieval_quality(validation: dict, docs: List[dict] = None) -> float:
    docs = docs or []
    doc_score = _doc_evidence_score(docs)
    if not validation:
        return doc_score if docs else 0.5

    relevance = float(validation.get("relevance_score", 0.5))
    coverage = float(validation.get("coverage_score", 0.5))
    val_quality = (relevance + coverage) / 2.0

    return (0.5 * val_quality + 0.5 * doc_score) if docs else val_quality


def _hallucination_risk(reflection: dict) -> float:
    if not reflection:
        return 0.5

    hallucinations = reflection.get("hallucinations", [])
    unsupported = reflection.get("unsupported_claims", [])
    is_supported = reflection.get("is_supported", True)

    if is_supported and not hallucinations and not unsupported:
        return 0.0

    risk = 0.0
    if hallucinations:
        risk += 0.40 + min(0.4, 0.10 * len(hallucinations))
    if unsupported:
        risk += min(0.25, 0.08 * len(unsupported))

    if is_supported and risk > 0.20:
        risk = 0.20
    elif not is_supported and risk < 0.40:
        risk = 0.40

    return min(1.0, risk)


def _citation_quality(citations: List[str], docs: List[dict], reflection: dict) -> float:
    if not docs:
        return 0.0

    num_citations = len(citations)
    if num_citations == 0:
        is_supported = reflection.get("is_supported", True) if reflection else True
        return 0.2 if is_supported else 0.0

    unique_citations = len(set(citations))
    ratio = unique_citations / max(1, len(docs))
    return min(1.0, 0.4 + 0.6 * ratio)


def _blended_confidence(
    reflection_score: float,
    retrieval_quality: float,
    hallucination_risk: float,
    citation_quality: float,
    retry_count: int = 0,
    local_retrieval_failed: bool = False,
) -> float:
    score = (
        settings.confidence_weight_reflection * reflection_score
        + settings.confidence_weight_retrieval * retrieval_quality
        + settings.confidence_weight_hallucination * (1.0 - hallucination_risk)
        + settings.confidence_weight_citation * citation_quality
    )

    # Query evidence adjustments
    if retry_count > 0:
        score *= max(0.85, 1.0 - 0.05 * retry_count)

    if local_retrieval_failed:
        score = min(settings.web_search_confidence_cap, score)

    return max(0.0, min(1.0, score))


def _confidence_level(score: float) -> str:
    if score >= settings.confidence_high_threshold:
        return "high"
    if score >= settings.confidence_medium_threshold:
        return "medium"
    return "low"


def _build_reason(retrieval_quality, hallucination_risk, reflection_score, citation_quality) -> str:
    parts = []
    if hallucination_risk > 0.3:
        parts.append(f"hallucination risk elevated ({hallucination_risk:.2f})")
    if retrieval_quality < 0.5:
        parts.append(f"retrieval quality low ({retrieval_quality:.2f})")
    if citation_quality < 0.4:
        parts.append(f"weak citation grounding ({citation_quality:.2f})")
    if reflection_score < 0.5:
        parts.append(f"reflection score low ({reflection_score:.2f})")
    if not parts:
        return "All sub-scores within acceptable range."
    return "; ".join(parts)


async def confidence_node(state: GraphState) -> GraphState:
    validation = state.get("validation", {})
    reflection = state.get("reflection", {})
    citations = state.get("citations", [])
    docs = state.get("reranked_docs") or state.get("merged_docs") or []
    retry_count = state.get("retry_count", 0)
    local_retrieval_failed = state.get("local_retrieval_failed", False)

    retrieval_quality = _retrieval_quality(validation, docs)
    hallucination_risk = _hallucination_risk(reflection)

    overall_score = float(reflection.get("overall_score", 0.85)) if reflection else 0.85
    completeness_score = float(reflection.get("completeness_score", 0.85)) if reflection else 0.85
    reflection_score = 0.6 * overall_score + 0.4 * completeness_score

    citation_quality = _citation_quality(citations, docs, reflection)

    confidence_score = _blended_confidence(
        reflection_score,
        retrieval_quality,
        hallucination_risk,
        citation_quality,
        retry_count=retry_count,
        local_retrieval_failed=local_retrieval_failed,
    )
    confidence_level = _confidence_level(confidence_score)
    reason = _build_reason(retrieval_quality, hallucination_risk, reflection_score, citation_quality)

    confidence = {
        "confidence_score": round(confidence_score, 3),
        "hallucination_risk": round(hallucination_risk, 3),
        "retrieval_quality": round(retrieval_quality, 3),
        "reflection_score": round(reflection_score, 3),
        "citation_quality": round(citation_quality, 3),
        "num_sources": len(citations),
        "confidence_level": confidence_level,
        "reason": reason,
    }

    logger.info(
        "confidence_estimated",
        extra={
            "question": state.get("question"),
            "confidence_score": confidence["confidence_score"],
            "confidence_level": confidence_level,
            "hallucination_risk": confidence["hallucination_risk"],
        },
    )

    return {**state, "confidence": confidence}

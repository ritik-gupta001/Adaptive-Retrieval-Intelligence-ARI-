
from typing import Any, Dict, List, Literal, Optional, TypedDict


class Document(TypedDict, total=False):
    content: str
    source: str
    score: float
    strategy: str          # which retriever produced this doc
    metadata: Dict[str, Any]


class QueryAttributes(TypedDict, total=False):
    intent: Literal["factual", "comparison", "research", "latest", "other"]
    complexity: Literal["simple", "medium", "complex"]
    freshness_needed: bool
    requires_multiple_sources: bool
    requires_reasoning: bool
    requires_comparison: bool
    domain: str
    entities: List[str]
    keywords: List[str]
    is_ambiguous: bool          # drives the future clarification node
    reasoning: str


class ConfidenceReport(TypedDict, total=False):
    confidence_score: float          # blended final score, 0-1
    hallucination_risk: float        # 0 (none) - 1 (severe), from reflection
    retrieval_quality: float         # 0-1, from context validation
    reflection_score: float          # 0-1, from reflection agent
    citation_quality: float          # 0-1, are citations well-grounded
    num_sources: int
    confidence_level: Literal["low", "medium", "high"]
    reason: str


class ReflectionReport(TypedDict, total=False):
    is_supported: bool
    hallucinations: List[str]
    unsupported_claims: List[str]
    missing_information: str
    incorrect_reasoning: List[str]
    completeness_score: float
    overall_score: float
    should_retry: bool
    reasoning: str


class ValidationReport(TypedDict, total=False):
    is_relevant: bool
    relevance_score: float
    coverage_score: float
    missing_information: str
    has_duplicates: bool
    issues: List[str]
    recommendation: Literal["proceed", "rewrite", "change_strategy", "ask_clarification"]


class GraphState(TypedDict, total=False):
    # --- input / identity ---
    question: str
    original_question: str
    conversation_id: str
    run_id: str

    # --- query understanding ---
    attributes: QueryAttributes

    # --- clarification ---
    clarification_needed: bool
    clarification_question: str

    # --- routing ---
    strategies: List[str]
    strategy_reasoning: str
    router_log: List[str]            # rule-hints vs llm-choice, for observability

    # --- retrieval ---
    retrieved_docs: List[Document]
    merged_docs: List[Document]
    reranked_docs: List[Document]

    # --- validation ---
    validation: ValidationReport

    # --- generation ---
    answer: str
    citations: List[str]

    # --- reflection + confidence ---
    reflection: ReflectionReport
    confidence: ConfidenceReport

    # --- control flow ---
    retry_count: int
    local_retrieval_failed: bool
    issues_log: List[str]
    final: bool

    # --- memory persistence ---
    short_term_summary: Optional[str]    # rolling summary the checkpointer persists
    long_term_context: Optional[Dict[str, Any]]  # pulled from the Store at start of turn


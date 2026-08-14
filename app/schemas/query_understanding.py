"""
Boundary schema for the Query Understanding Agent's LLM output.

Why this exists separately from app/graph/state.py's QueryAttributes
TypedDict: the TypedDict describes what *flows through the graph*; this
Pydantic model is what we validate the *raw, untrusted LLM JSON* against
before it's allowed to become part of GraphState. LLMs occasionally drop a
field, use the wrong type, or invent an enum value outside what we asked
for — this is where that gets caught, with a clear error, instead of a
KeyError three nodes later in the reranker.
"""
from typing import List, Literal

from pydantic import BaseModel, Field, field_validator


class QueryUnderstandingOutput(BaseModel):
    intent: Literal["factual", "comparison", "research", "latest", "other"]
    complexity: Literal["simple", "medium", "complex"]
    freshness_needed: bool
    requires_multiple_sources: bool
    requires_reasoning: bool = False
    requires_comparison: bool = False
    domain: str = Field(default="general")
    entities: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    reasoning: str = Field(default="")

    @field_validator("domain")
    @classmethod
    def _domain_not_empty(cls, v: str) -> str:
        return v.strip() or "general"

    @field_validator("keywords")
    @classmethod
    def _keywords_nonempty_for_non_trivial(cls, v: List[str]) -> List[str]:
        # Not a hard failure — just normalize. Some genuinely trivial
        # queries ("hi") legitimately have no useful keywords.
        return [k.strip() for k in v if k.strip()]

    def is_ambiguous(self) -> bool:
        """Heuristic used by the node to flag clarification candidates.
        Kept here (not duplicated in the node) since it's a property of the
        classification output itself, not of orchestration logic.

        Ambiguous if: no entities extracted AND intent fell through to
        'other' AND it wasn't classified as a simple/trivial query.
        """
        return (
            len(self.entities) == 0
            and self.intent == "other"
            and self.complexity != "simple"
        )

"""
Boundary schema for the Context Validator's LLM output. Mirrors the
ValidationReport TypedDict in app/graph/state.py — this is the
validated-then-trusted version of that shape.
"""
from typing import List, Literal

from pydantic import BaseModel, Field, field_validator


class ValidationOutput(BaseModel):
    is_relevant: bool
    relevance_score: float = Field(ge=0.0, le=1.0)
    coverage_score: float = Field(ge=0.0, le=1.0)
    missing_information: str = Field(default="")
    has_duplicates: bool = False
    issues: List[str] = Field(default_factory=list)
    recommendation: Literal["proceed", "rewrite", "change_strategy", "ask_clarification"]

    @field_validator("issues")
    @classmethod
    def _clean_issues(cls, v: List[str]) -> List[str]:
        return [i.strip() for i in v if i.strip()]

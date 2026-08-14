"""
Boundary schema for the Reflection Agent's LLM output. Mirrors the
ReflectionReport TypedDict in app/graph/state.py.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class ReflectionOutput(BaseModel):
    is_supported: bool
    hallucinations: List[str] = Field(default_factory=list)
    unsupported_claims: List[str] = Field(default_factory=list)
    missing_information: str = Field(default="")
    incorrect_reasoning: List[str] = Field(default_factory=list)
    completeness_score: float = Field(..., ge=0.0, le=1.0)
    overall_score: float = Field(..., ge=0.0, le=1.0)
    should_retry: bool = Field(default=False)
    reasoning: str = Field(default="")

    @field_validator("hallucinations", "unsupported_claims", "incorrect_reasoning", mode="before")
    @classmethod
    def _clean_lists(cls, v: Any) -> List[str]:
        if isinstance(v, list):
            return [str(item).strip() for item in v if str(item).strip()]
        if isinstance(v, str) and v.strip():
            return [v.strip()]
        return []

    @field_validator("should_retry", mode="before")
    @classmethod
    def _parse_should_retry(cls, v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes", "y")
        return False

    @model_validator(mode="before")
    @classmethod
    def _ensure_should_retry(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # If should_retry is missing, compute it from is_supported, overall_score, and hallucinations
            if "should_retry" not in data or data["should_retry"] is None:
                is_sup = data.get("is_supported", True)
                score = data.get("overall_score", 0.85)
                halls = data.get("hallucinations", [])
                data["should_retry"] = (not is_sup) or (score < 0.65) or bool(halls)
        return data


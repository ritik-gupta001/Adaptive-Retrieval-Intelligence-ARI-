"""
Boundary schema for the Adaptive Router's LLM output.

Accepts dynamic strategy lists and validates strategy names against configured strategies.
"""

from typing import List

from pydantic import BaseModel, Field, field_validator


class RouterOutput(BaseModel):
    strategies: List[str] = Field(default_factory=list)
    reasoning: str = Field(default="")

    @field_validator("strategies")
    @classmethod
    def _dedupe_preserve_order(cls, v: List[str]) -> List[str]:
        seen = set()
        out = []
        for s in v:
            s_norm = s.strip().lower()
            if s_norm and s_norm not in seen:
                seen.add(s_norm)
                out.append(s_norm)
        return out

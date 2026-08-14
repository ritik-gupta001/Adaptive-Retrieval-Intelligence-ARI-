from typing import List

from pydantic import BaseModel, Field, field_validator


class MultiQueryOutput(BaseModel):
    queries: List[str] = Field(default_factory=list)

    @field_validator("queries")
    @classmethod
    def _clean(cls, v: List[str]) -> List[str]:
        return [q.strip() for q in v if q.strip()]

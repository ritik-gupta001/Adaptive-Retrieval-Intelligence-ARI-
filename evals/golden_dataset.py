
import json
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

GOLDEN_PATH = Path(__file__).parent / "golden_queries.jsonl"


class GoldenRecord(BaseModel):
    question: str
    ground_truth: str
    contexts: List[str] = Field(min_length=1)
    expected_intent: Optional[str] = None
    expected_strategy: Optional[str] = None

    @field_validator("question", "ground_truth")
    @classmethod
    def _not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question and ground_truth must not be empty")
        return v

    @field_validator("contexts")
    @classmethod
    def _contexts_not_empty_strings(cls, v: List[str]) -> List[str]:
        cleaned = [c.strip() for c in v if c.strip()]
        if not cleaned:
            raise ValueError("contexts must contain at least one non-empty string")
        return cleaned


def load_golden_dataset(path: Path = GOLDEN_PATH) -> List[GoldenRecord]:
    """Load and validate every record in the JSONL file.
    Raises ValueError with the line number if any record fails validation."""
    if not path.exists():
        raise FileNotFoundError(
            f"Golden dataset not found at {path}. "
            "Populate evals/golden_queries.jsonl before running evaluation."
        )

    records = []
    with open(path) as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                records.append(GoldenRecord.model_validate(raw))
            except (json.JSONDecodeError, Exception) as exc:
                raise ValueError(
                    f"golden_queries.jsonl line {i} is invalid: {exc}"
                ) from exc

    if not records:
        raise ValueError("golden_queries.jsonl is empty — add at least one record")

    return records

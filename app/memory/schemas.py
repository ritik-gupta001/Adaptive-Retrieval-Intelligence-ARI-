"""
Typed models for everything written to the LangGraph Store.

Three namespaces, each a separate key pattern:
  - user_preferences:{conversation_id}   → UserPreferences
  - conversation_summary:{conversation_id} → ConversationSummary
  - rewrite_cache:{query_hash}            → RewriteCache

Why Pydantic here (not TypedDict): Store values are serialized to JSON
and deserialized back — Pydantic's model_dump / model_validate gives us
safe round-trip serialization with field-level validation, catching drift
between what was written and what a later schema version expects to read.
"""
from datetime import datetime, timezone
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class UserPreferences(BaseModel):
    """Accumulated from successful interactions — never explicitly set by
    the user, inferred from what worked (which strategy, what domain)."""
    preferred_strategies: List[str] = Field(default_factory=list)
    preferred_domains: List[str] = Field(default_factory=list)
    language: str = "en"
    last_updated: str = Field(default_factory=_now)


class TurnRecord(BaseModel):
    """One completed turn: question asked, answer given, how confident."""
    question: str
    answer: str
    citations: List[str] = Field(default_factory=list)
    strategies_used: List[str] = Field(default_factory=list)
    confidence_score: float = 0.0
    timestamp: str = Field(default_factory=_now)


class ConversationSummary(BaseModel):
    """Rolling log of the last N turns for this conversation thread.
    Injected into state.long_term_context so the LLM can refer back to
    earlier turns without the whole history inflating every prompt."""
    conversation_id: str
    turns: List[TurnRecord] = Field(default_factory=list)
    last_updated: str = Field(default_factory=_now)

    MAX_TURNS: int = 10  # keep only the last 10 turns

    def add_turn(self, turn: TurnRecord) -> None:
        self.turns.append(turn)
        if len(self.turns) > self.MAX_TURNS:
            self.turns = self.turns[-self.MAX_TURNS :]
        self.last_updated = _now()


class RewriteCache(BaseModel):
    """Successful rewrites — if the same (or very similar) question comes
    in again we can skip the rewrite LLM call and jump straight to the
    rewrite that previously worked."""
    original_question: str
    successful_rewrite: str
    strategies_that_worked: List[str] = Field(default_factory=list)
    success_count: int = 1
    last_used: str = Field(default_factory=_now)

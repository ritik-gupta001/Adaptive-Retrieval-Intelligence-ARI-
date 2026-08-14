"""
Long-term Store wrapper.
Provides typed, schema-validated storage methods for user preferences, conversation summaries, and query rewrite caching.
"""

import hashlib
import json
from typing import Optional

from app.core.logging import get_logger
from app.memory.schemas import (
    ConversationSummary,
    RewriteCache,
    TurnRecord,
    UserPreferences,
)

logger = get_logger(__name__)


class ARIStore:
    """
    Typed wrapper around any LangGraph BaseStore-compatible backend.
    Pass `backend=None` to use InMemoryStore (default for dev/test).
    """

    def __init__(self, backend=None):
        if backend is not None:
            self._store = backend
        else:
            from langgraph.store.memory import InMemoryStore
            self._store = InMemoryStore()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    async def _get(self, namespace: tuple, key: str) -> Optional[dict]:
        try:
            item = await self._store.aget(namespace, key)
            return item.value if item else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("store_get_failed", extra={"namespace": namespace, "key": key, "error": str(exc)})
            return None

    async def _put(self, namespace: tuple, key: str, value: dict) -> None:
        try:
            await self._store.aput(namespace, key, value)
        except Exception as exc:  # noqa: BLE001
            logger.error("store_put_failed", extra={"namespace": namespace, "key": key, "error": str(exc)})

    # ------------------------------------------------------------------ #
    # User Preferences
    # ------------------------------------------------------------------ #

    async def get_preferences(self, conversation_id: str) -> Optional[UserPreferences]:
        raw = await self._get(("preferences",), conversation_id)
        if raw is None:
            return None
        try:
            return UserPreferences.model_validate(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("preferences_parse_failed", extra={"error": str(exc)})
            return None

    async def save_preferences(self, conversation_id: str, prefs: UserPreferences) -> None:
        await self._put(("preferences",), conversation_id, prefs.model_dump())

    async def update_preferences_from_turn(
        self, conversation_id: str, strategies: list, domain: str
    ) -> None:
        """Infer preferences from what worked this turn — accumulated over
        time, not set explicitly by the user."""
        prefs = await self.get_preferences(conversation_id) or UserPreferences()
        for s in strategies:
            if s not in prefs.preferred_strategies:
                prefs.preferred_strategies.append(s)
        if domain and domain != "general" and domain not in prefs.preferred_domains:
            prefs.preferred_domains.append(domain)
        await self.save_preferences(conversation_id, prefs)

    # ------------------------------------------------------------------ #
    # Conversation Summary (rolling turn log)
    # ------------------------------------------------------------------ #

    async def get_summary(self, conversation_id: str) -> Optional[ConversationSummary]:
        raw = await self._get(("conversations",), conversation_id)
        if raw is None:
            return None
        try:
            return ConversationSummary.model_validate(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("summary_parse_failed", extra={"error": str(exc)})
            return None

    async def append_turn(self, conversation_id: str, turn: TurnRecord) -> None:
        summary = await self.get_summary(conversation_id) or ConversationSummary(
            conversation_id=conversation_id
        )
        summary.add_turn(turn)
        await self._put(("conversations",), conversation_id, summary.model_dump())

    # ------------------------------------------------------------------ #
    # Rewrite Cache
    # ------------------------------------------------------------------ #

    @staticmethod
    def _query_hash(question: str) -> str:
        """Stable hash key for a question — normalized before hashing so
        minor whitespace/case differences don't create separate cache entries."""
        normalized = " ".join(question.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    async def get_rewrite(self, question: str) -> Optional[RewriteCache]:
        key = self._query_hash(question)
        raw = await self._get(("rewrites",), key)
        if raw is None:
            return None
        try:
            return RewriteCache.model_validate(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("rewrite_cache_parse_failed", extra={"error": str(exc)})
            return None

    async def save_rewrite(
        self, original: str, rewrite: str, strategies: list
    ) -> None:
        key = self._query_hash(original)
        existing = await self.get_rewrite(original)
        if existing:
            existing.success_count += 1
            existing.successful_rewrite = rewrite
            existing.strategies_that_worked = strategies
            await self._put(("rewrites",), key, existing.model_dump())
        else:
            cache = RewriteCache(
                original_question=original,
                successful_rewrite=rewrite,
                strategies_that_worked=strategies,
            )
            await self._put(("rewrites",), key, cache.model_dump())


# Module-level singleton used by the memory nodes. Tests swap this out
# via monkeypatch to inject a fresh InMemoryStore-backed ARIStore.
_store: Optional[ARIStore] = None


def get_store() -> ARIStore:
    global _store
    if _store is None:
        _store = ARIStore()
    return _store

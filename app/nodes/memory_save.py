"""
Node: Memory Save.
Runs AFTER finalize. Persists what was learned this turn to the Store.

Three writes, each independent (one failing doesn't abort the others):
  1. Conversation turn record    → ConversationSummary (rolling log)
  2. User preferences update     → inferred from strategies that worked
  3. Rewrite cache entry         → if this turn involved a rewrite AND the
                                   final confidence was high enough to trust

Write threshold: we only write to the Store when confidence is "medium"
or "high" — storing a failed/low-confidence turn as a "successful"
interaction would poison future context with bad examples.

Failure posture: same as memory_load — Store write failures never block
the pipeline. The response has already been assembled by finalize; this
node only side-effects the Store.
"""
from app.config.settings import settings
from app.core.logging import get_logger
from app.graph.state import GraphState
from app.memory.schemas import TurnRecord
from app.memory.store import get_store

logger = get_logger(__name__)

_WRITE_CONFIDENCE_LEVELS = {"medium", "high"}



async def memory_save_node(state: GraphState) -> GraphState:
    conversation_id = state.get("conversation_id", "default")
    question = state.get("question", "")
    original_question = state.get("original_question", question)
    answer = state.get("answer", "")
    citations = state.get("citations", [])
    strategies = state.get("strategies", [])
    confidence = state.get("confidence", {})
    attributes = state.get("attributes", {})
    retry_count = state.get("retry_count", 0)

    confidence_level = confidence.get("confidence_level", "low")
    confidence_score = confidence.get("confidence_score", 0.0)

    if confidence_level not in _WRITE_CONFIDENCE_LEVELS:
        logger.info(
            "memory_save_skipped_low_confidence",
            extra={
                "conversation_id": conversation_id,
                "confidence_level": confidence_level,
            },
        )
        return state

    store = get_store()

    # 1 — Append turn to conversation summary
    try:
        turn = TurnRecord(
            question=original_question,
            answer=answer,
            citations=citations,
            strategies_used=strategies,
            confidence_score=confidence_score,
        )
        await store.append_turn(conversation_id, turn)
        logger.info(
            "memory_turn_saved",
            extra={"conversation_id": conversation_id, "confidence_score": confidence_score},
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("memory_save_turn_failed", extra={"error": str(exc)})

    # 2 — Update inferred user preferences
    try:
        domain = attributes.get("domain", "general")
        await store.update_preferences_from_turn(conversation_id, strategies, domain)
    except Exception as exc:  # noqa: BLE001
        logger.error("memory_save_preferences_failed", extra={"error": str(exc)})

    # 3 — Cache the rewrite if this turn was a successful retry
    if retry_count > 0 and confidence_level == "high":
        try:
            await store.save_rewrite(
                original=original_question,
                rewrite=question,  # current question = the rewritten version
                strategies=strategies,
            )
            logger.info(
                "memory_rewrite_cached",
                extra={
                    "original": original_question,
                    "rewrite": question,
                    "strategies": strategies,
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("memory_save_rewrite_failed", extra={"error": str(exc)})

    return state

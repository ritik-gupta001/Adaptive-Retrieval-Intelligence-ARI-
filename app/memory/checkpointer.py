"""
Checkpointer factory.

Returns the right LangGraph checkpointer for the configured backend:
  - "memory"   → MemorySaver   (dev/test, in-process, lost on restart)
  - "sqlite"   → SqliteSaver   (local prod, single-process, persists to file)
  - "postgres" → AsyncPostgresSaver (full prod, requires DATABASE_URL)

The checkpointer is passed to graph.compile(checkpointer=...) in
build_graph.py. It's completely transparent to every node — nodes never
import or interact with it directly. What it gives the pipeline:
  1. Resumability: if a node raises, the next invocation with the same
     thread_id picks up from the last successful checkpoint rather than
     re-running the whole pipeline.
  2. Multi-turn threads: subsequent questions in the same conversation
     share state history via thread_id in the invocation RunnableConfig.
  3. LangSmith replay: checkpointed supersteps are what LangSmith uses
     to reconstruct graph execution traces for debugging.

Why async for postgres but sync for memory/sqlite: MemorySaver and
SqliteSaver are synchronous (no I/O) and LangGraph wraps them in a
thread-executor automatically when used in an async graph. AsyncPostgresSaver
is a proper asyncpg-backed saver that needs an async setup call
(await saver.setup()) before first use.
"""
from functools import lru_cache
from typing import Union

from app.config.settings import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def get_checkpointer():
    """
    Returns an appropriate checkpointer for the configured backend.
    Cached per process — checkpointers hold internal state (connection
    pools, in-memory dicts) that should not be reconstructed per request.
    """
    backend = settings.checkpointer_backend
    logger.info("checkpointer_initializing", extra={"backend": backend})

    if backend == "memory":
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()

    if backend == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:
            raise ImportError(
                "langgraph-checkpoint-sqlite is not installed. "
                "Run: pip install langgraph-checkpoint-sqlite"
            ) from exc
        conn_str = settings.checkpointer_conn_str or "./data/checkpoints.db"
        return SqliteSaver.from_conn_string(conn_str)

    if backend == "postgres":
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        except ImportError as exc:
            raise ImportError(
                "langgraph-checkpoint-postgres is not installed. "
                "Run: pip install langgraph-checkpoint-postgres"
            ) from exc
        if not settings.checkpointer_conn_str:
            raise ValueError(
                "checkpointer_backend='postgres' requires "
                "CHECKPOINTER_CONN_STR to be set in .env"
            )
        # Note: AsyncPostgresSaver requires `await saver.setup()` before first
        # use. The FastAPI lifespan handler in app/main.py is responsible for
        # calling that — not this factory.
        return AsyncPostgresSaver.from_conn_string(settings.checkpointer_conn_str)

    raise ValueError(
        f"Unknown checkpointer_backend: '{backend}'. "
        f"Valid options: memory, sqlite, postgres"
    )

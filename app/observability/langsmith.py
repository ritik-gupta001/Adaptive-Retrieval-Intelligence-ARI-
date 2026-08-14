"""
LangSmith tracing integration.
Provides structured run tagging and tracing metadata attachment for LangSmith observability dashboards.

run_context() is an async context manager used by the streaming endpoint
to open a named parent run that all node-level child runs attach to,
giving a clean "this is one user request" boundary in the LangSmith UI.
"""
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from app.config.settings import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_langsmith_available = False
_client = None


def configure_langsmith() -> None:
    """Call once at startup (from app/main.py lifespan). Sets env vars
    LangChain reads automatically, and pre-checks SDK availability so we
    know at startup whether tagging will actually work, not at first call."""
    global _langsmith_available, _client

    if not settings.langsmith_tracing_enabled:
        logger.info("langsmith_tracing_disabled")
        return

    if not settings.langsmith_api_key:
        logger.warning(
            "langsmith_tracing_enabled_but_no_api_key — tracing will not work"
        )
        return

    # LangChain reads these env vars automatically — set them here so they're
    # guaranteed to be present even if the user only set our typed settings.
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project

    try:
        from langsmith import Client
        _client = Client(api_key=settings.langsmith_api_key)
        _langsmith_available = True
        logger.info(
            "langsmith_configured",
            extra={"project": settings.langsmith_project},
        )
    except ImportError:
        logger.warning("langsmith_sdk_not_installed — pip install langsmith")
    except Exception as exc:  # noqa: BLE001
        logger.warning("langsmith_init_failed", extra={"error": str(exc)})


def tag_run(run_id: Optional[str], tags: Dict[str, Any]) -> None:
    """Attach structured metadata to the current LangSmith run.
    Called from nodes that have domain-specific fields worth tagging
    (e.g. router → strategy chosen, confidence_node → confidence_score).

    Degrades silently when LangSmith is unavailable — no try/except needed
    at call sites; this function absorbs all SDK errors internally."""
    if not _langsmith_available or not _client or not run_id:
        return
    try:
        _client.update_run(run_id, extra={"structured_metadata": tags})
    except Exception as exc:  # noqa: BLE001
        logger.debug("langsmith_tag_run_failed", extra={"error": str(exc)})


@asynccontextmanager
async def run_context(name: str, inputs: Dict[str, Any]):
    """Async context manager that opens a named LangSmith parent run.
    Yields the run_id so child operations can reference it.
    Used by the streaming endpoint to create a clean per-request boundary
    in the trace UI.

    If LangSmith is unavailable, yields a placeholder run_id (uuid) so
    call sites don't need to handle the None case."""
    import uuid
    run_id = str(uuid.uuid4())

    if not _langsmith_available or not _client:
        yield run_id
        return

    try:
        from langsmith.run_trees import RunTree
        run = RunTree(
            name=name,
            run_type="chain",
            inputs=inputs,
            project_name=settings.langsmith_project,
        )
        await run.apost()
        yield run.id
        run.end(outputs={"status": "completed"})
        await run.apost()
    except Exception as exc:  # noqa: BLE001
        logger.debug("langsmith_run_context_failed", extra={"error": str(exc)})
        yield run_id


def is_available() -> bool:
    return _langsmith_available

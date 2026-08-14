"""
FastAPI application entrypoint.

Handles application startup and shutdown lifespan management:
  1. Pre-warms the compiled LangGraph pipeline.
  2. Initializes session checkpointer storage.
  3. Initializes memory store backend.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import health, query, stream
from app.config.settings import settings
from app.core.logging import configure_logging, get_logger
from app.observability.langsmith import configure_langsmith
from app.observability.middleware import ObservabilityMiddleware

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    logger.info("ari_startup_begin")

    # 0. LangSmith tracing
    configure_langsmith()

    # 1. Pre-warm the compiled graph (validates edges, node wiring)
    try:
        from app.graph.build_graph import get_compiled_graph
        get_compiled_graph()
        logger.info("graph_compiled_ok")
    except Exception as exc:  # noqa: BLE001
        logger.error("graph_compile_failed_at_startup", extra={"error": str(exc)})

    # 2. Checkpointer async setup (required for AsyncPostgresSaver)
    try:
        from app.memory.checkpointer import get_checkpointer
        checkpointer = get_checkpointer()
        if hasattr(checkpointer, "setup"):
            await checkpointer.setup()
            logger.info("checkpointer_setup_ok")
    except Exception as exc:  # noqa: BLE001
        logger.warning("checkpointer_setup_failed", extra={"error": str(exc)})

    # 3. Store init
    try:
        from app.memory.store import get_store
        get_store()
        logger.info("store_init_ok")
    except Exception as exc:  # noqa: BLE001
        logger.warning("store_init_failed", extra={"error": str(exc)})

    # 4. Pre-warm BM25 Knowledge Index (12,403 chunks loaded into RAM ONCE at startup)
    try:
        from app.retrievers.hybrid_search import get_global_bm25_index
        get_global_bm25_index()
        logger.info("bm25_index_prewarmed_ok")
    except Exception as exc:  # noqa: BLE001
        logger.warning("bm25_prewarm_failed", extra={"error": str(exc)})

    # 5. Pre-warm BGE CrossEncoder if configured
    if settings.reranker_provider == "bge":
        try:
            from app.nodes.rerank import _get_reranker
            _get_reranker()
            logger.info("bge_reranker_prewarmed_ok")
        except Exception as exc:  # noqa: BLE001
            logger.error("bge_reranker_prewarm_failed_at_startup", extra={"error": str(exc)})
            raise

    logger.info("ari_startup_complete")
    yield

    # --- shutdown ---
    logger.info("ari_shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Adaptive Retrieval Intelligence (ARI)",
        description=(
            "Production-grade adaptive RAG platform. Dynamically selects "
            "the best retrieval strategy per query, with self-reflection, "
            "confidence estimation, and multi-turn memory."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # Observability middleware (before CORS so it times the full request)
    app.add_middleware(ObservabilityMiddleware)

    # CORS Configuration
    # WHY: Browsers reject CORS requests with allow_origins=["*"] when allow_credentials=True.
    # If ALLOW_ORIGINS is unset or wildcard ("*"), allow_credentials is set to False with a
    # warning logged at startup. When explicit origins are provided, allow_credentials=True is allowed.
    raw_origins = os.getenv("ALLOW_ORIGINS", "*").strip()
    if not raw_origins or raw_origins == "*":
        allow_origins = ["*"]
        allow_credentials = False
        logger.warning("CORS running in permissive/dev mode: allow_origins=['*'] with allow_credentials=False")
    else:
        allow_origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
        allow_credentials = True

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )


    # Routers
    app.include_router(health.router, tags=["health"])
    app.include_router(query.router, tags=["query"])
    app.include_router(stream.router, tags=["stream"])

    return app


app = create_app()

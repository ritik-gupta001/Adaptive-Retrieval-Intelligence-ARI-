"""
GET /health  — liveness + readiness combined.
GET /graph   — graph structure for frontend live visualization.
"""
from datetime import datetime, timezone

from fastapi import APIRouter

from app.api.schemas import GraphStructureResponse, HealthResponse
from app.config.settings import settings
from app.graph.build_graph import build_graph, get_compiled_graph

router = APIRouter()


@router.get("/")
async def root():
    return {
        "name": "Adaptive Retrieval Intelligence (ARI) API",
        "status": "online",
        "health": "/health",
        "docs": "/docs",
    }


@router.get("/health", response_model=HealthResponse)

async def health() -> HealthResponse:
    try:
        get_compiled_graph()
        graph_compiled = True
    except Exception:  # noqa: BLE001
        graph_compiled = False

    return HealthResponse(
        status="ok" if graph_compiled else "degraded",
        timestamp=datetime.now(timezone.utc).isoformat(),
        graph_compiled=graph_compiled,
        checkpointer_backend=settings.checkpointer_backend,
        store_backend=settings.store_backend,
    )


@router.get("/graph", response_model=GraphStructureResponse)
async def graph_structure() -> GraphStructureResponse:
    compiled = get_compiled_graph()
    g = compiled.get_graph()

    nodes = [n for n in g.nodes.keys() if n not in ("__start__", "__end__")]
    edges = [
        {"source": e.source, "target": e.target}
        for e in g.edges
        if e.source not in ("__start__",) and e.target not in ("__end__",)
    ]

    return GraphStructureResponse(nodes=nodes, edges=edges)

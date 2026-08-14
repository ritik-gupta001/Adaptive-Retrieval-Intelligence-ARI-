"""
POST /query/stream
Server-Sent Events streaming endpoint. Emits one SSE per node completion
so the frontend can update the live graph visualization and show progress
rather than waiting for the full pipeline to finish.

"""
import json
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.api.schemas import QueryRequest
from app.config.settings import settings
from app.core.logging import current_run_id, get_logger
from app.graph.build_graph import get_compiled_graph, get_recursion_limit
from app.security.auth import check_rate_limit, verify_api_key
from app.security.guardrails import validate_input_security

logger = get_logger(__name__)
router = APIRouter()

# Nodes we want to surface in the stream (not all internal supersteps)
_SURFACED_NODES = {
    "memory_load",
    "query_understanding",
    "adaptive_router",
    "retrieve",
    "document_merge",
    "rerank",
    "context_validate",
    "generate",
    "reflect",
    "confidence_score",
    "query_rewrite",
    "finalize",
    "memory_save",
}

_NODE_LABELS = {
    "memory_load": "Loading memory",
    "query_understanding": "Understanding query",
    "adaptive_router": "Selecting strategy",
    "retrieve": "Retrieving documents",
    "document_merge": "Merging documents",
    "rerank": "Reranking results",
    "context_validate": "Validating context",
    "generate": "Generating answer",
    "reflect": "Reflecting on answer",
    "confidence_score": "Estimating confidence",
    "query_rewrite": "Rewriting query",
    "finalize": "Finalizing response",
    "memory_save": "Saving to memory",
}


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def _event_generator(req: QueryRequest) -> AsyncGenerator[str, None]:
    conversation_id = req.resolved_conversation_id()
    thread_id = req.resolved_thread_id(conversation_id)
    run_id = str(uuid.uuid4())
    current_run_id.set(run_id)

    graph = get_compiled_graph()
    initial_state = {
        "question": req.question,
        "original_question": req.question,
        "conversation_id": conversation_id,
        "run_id": run_id,
        "retry_count": 0,
        "issues_log": [],
    }
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": get_recursion_limit(),
    }

    final_state = {}

    try:
        async for event in graph.astream_events(initial_state, config, version="v2"):
            event_name = event.get("event", "")
            node_name = event.get("name", "")

            # Node completed — emit progress event
            if event_name == "on_chain_end" and node_name in _SURFACED_NODES:
                output = event.get("data", {}).get("output", {}) or {}

                # Accumulate final_state incrementally from each node's output
                if isinstance(output, dict):
                    final_state.update(output)

                payload = {
                    "event": "node_complete",
                    "node": node_name,
                    "label": _NODE_LABELS.get(node_name, node_name),
                    "data": {},
                }

                # Surface lightweight per-node summaries, not full state
                if node_name == "query_understanding" and output.get("attributes"):
                    payload["data"]["intent"] = output["attributes"].get("intent")
                    payload["data"]["complexity"] = output["attributes"].get("complexity")

                if node_name == "adaptive_router":
                    payload["data"]["strategies"] = output.get("strategies", [])
                    payload["data"]["clarification_needed"] = output.get("clarification_needed", False)

                if node_name == "retrieve":
                    payload["data"]["docs_retrieved"] = len(output.get("retrieved_docs") or [])

                if node_name == "context_validate":
                    validation = output.get("validation") or {}
                    payload["data"]["recommendation"] = validation.get("recommendation")

                if node_name == "generate":
                    answer = output.get("answer", "")
                    payload["data"]["answer_preview"] = answer[:200] if answer else ""

                if node_name == "reflect":
                    reflection = output.get("reflection") or {}
                    payload["data"]["is_supported"] = reflection.get("is_supported")
                    payload["data"]["should_retry"] = reflection.get("should_retry")

                if node_name == "confidence_score":
                    confidence = output.get("confidence") or {}
                    payload["data"]["confidence_level"] = confidence.get("confidence_level")
                    payload["data"]["confidence_score"] = confidence.get("confidence_score")

                if node_name == "query_rewrite":
                    payload["data"]["rewritten_question"] = output.get("question", "")
                    payload["data"]["retry_count"] = output.get("retry_count", 0)

                yield _sse(payload)

    except Exception as exc:  # noqa: BLE001
        logger.error("stream_error", extra={"error": str(exc)})
        yield _sse({"event": "error", "message": str(exc)})
        return

    # Final event — full response payload
    confidence_raw = final_state.get("confidence") or {}
    yield _sse({
        "event": "final",
        "answer": final_state.get("answer", ""),
        "citations": final_state.get("citations", []),
        "confidence": {
            "confidence_score": confidence_raw.get("confidence_score", 0.0),
            "confidence_level": confidence_raw.get("confidence_level", "low"),
            "hallucination_risk": confidence_raw.get("hallucination_risk", 0.0),
            "retrieval_quality": confidence_raw.get("retrieval_quality", 0.0),
            "reflection_score": confidence_raw.get("reflection_score", 0.0),
            "citation_quality": confidence_raw.get("citation_quality", 0.0),
            "num_sources": confidence_raw.get("num_sources", 0),
            "reason": confidence_raw.get("reason", ""),
        },
        "strategies_used": final_state.get("strategies", []),
        "retry_count": final_state.get("retry_count", 0),
        "conversation_id": conversation_id,
        "thread_id": thread_id,
        "clarification_needed": final_state.get("clarification_needed", False),
        "clarification_question": final_state.get("clarification_question"),
        "memory_context": final_state.get("long_term_context"),
    })


@router.post(
    "/query/stream",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def query_stream(req: QueryRequest, request: Request) -> StreamingResponse:
    is_safe, block_reason = validate_input_security(req.question)
    if not is_safe:
        logger.warning(
            "stream_blocked_by_guardrail",
            extra={"question": req.question[:100], "reason": block_reason},
        )
        raise HTTPException(status_code=400, detail=block_reason)

    return StreamingResponse(
        _event_generator(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx proxy buffering for SSE
        },
    )

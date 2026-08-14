#!/usr/bin/env python
"""
scripts/run_langsmith_traces.py
-------------------------------
Runs representative benchmark queries through the ARI pipeline with LangSmith tracing
enabled, including multi-turn memory testing.

Usage:
    python scripts/run_langsmith_traces.py

Requires:
    - LANGSMITH_TRACING_ENABLED=true in .env
    - LANGSMITH_API_KEY set in .env
"""
import asyncio
import os
import sys

# Ensure UTF-8 output encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.append(".")

from app.core.logging import configure_logging, current_run_id

configure_logging()

from app.config.settings import settings
from app.observability.langsmith import configure_langsmith, is_available
from app.graph.build_graph import get_compiled_graph
from app.core.logging import get_logger

logger = get_logger(__name__)

TRACE_QUERIES = [
    ("What is LangGraph?", "langsmith-trace-1"),
    ("Compare vector search and hybrid search for RAG.", "langsmith-trace-2"),
    ("What are the latest developments in agentic AI systems?", "langsmith-trace-3"),
    ("Explain how Reciprocal Rank Fusion works.", "langsmith-trace-4"),
    ("How does the reflection agent detect hallucinations in ARI?", "langsmith-trace-5"),
    ("In that case, how does state persistence work in it?", "langsmith-trace-1"),  # Multi-turn memory query using Turn 1 thread
]


async def run_query(graph, question: str, conv_id: str, idx: int) -> dict:
    import uuid, time
    run_id = f"trace-run-{idx}-{uuid.uuid4().hex[:8]}"
    current_run_id.set(run_id)

    initial_state = {
        "question": question,
        "original_question": question,
        "conversation_id": conv_id,
        "run_id": run_id,
        "retry_count": 0,
        "issues_log": [],
    }
    config = {
        "configurable": {"thread_id": f"thread-{conv_id}"},
        "recursion_limit": 50,
    }
    start = time.monotonic()
    try:
        final_state = await graph.ainvoke(initial_state, config)
        elapsed = round(time.monotonic() - start, 2)
        confidence = final_state.get("confidence") or {}
        reflection = final_state.get("reflection") or {}
        reranked = final_state.get("reranked_docs") or []
        retrieved = final_state.get("retrieved_docs") or []
        memory = final_state.get("long_term_context") or {}
        has_history = bool(memory.get("conversation_history"))
        return {
            "question": question,
            "answer_preview": (final_state.get("answer") or "")[:150],
            "strategies": final_state.get("strategies", []),
            "confidence_score": confidence.get("confidence_score", 0.0),
            "confidence_level": confidence.get("confidence_level", "unknown"),
            "reflection_is_supported": reflection.get("is_supported", True),
            "reflection_should_retry": reflection.get("should_retry", False),
            "reflection_overall_score": reflection.get("overall_score", 0.0),
            "retrieved_count": len(retrieved),
            "reranked_count": len(reranked),
            "memory_loaded": has_history,
            "latency_seconds": elapsed,
            "retry_count": final_state.get("retry_count", 0),
            "run_id": run_id,
            "error": None,
        }
    except Exception as exc:
        elapsed = round(time.monotonic() - start, 2)
        return {"question": question, "error": str(exc), "latency_seconds": elapsed, "run_id": run_id}


async def main():
    print("=" * 70)
    print("ARI LangSmith Observability & Memory Trace Runner")
    print("=" * 70)

    # Configure LangSmith
    configure_langsmith()
    if not is_available():
        print("\n[!] LangSmith is not available.")
        print("   Check LANGSMITH_TRACING_ENABLED=true and LANGSMITH_API_KEY in .env")
        print(f"   Current: LANGSMITH_TRACING_ENABLED={settings.langsmith_tracing_enabled}")
        print(f"   Key set: {bool(settings.langsmith_api_key)}")
        sys.exit(1)

    print(f"\n[OK] LangSmith tracing active -> project: '{settings.langsmith_project}'")
    print(f"  View traces at: https://smith.langchain.com/projects\n")

    graph = get_compiled_graph()

    print(f"Running {len(TRACE_QUERIES)} queries through the ARI pipeline...\n")
    all_successful = True
    for i, (query, conv_id) in enumerate(TRACE_QUERIES, start=1):
        print(f"[{i}/{len(TRACE_QUERIES)}] Query: {query} (conv_id={conv_id})")
        result = await run_query(graph, query, conv_id, i)
        if result.get("error"):
            all_successful = False
            print(f"  [FAIL] ERROR ({result['latency_seconds']}s): {result['error']}")
        else:
            print(f"  [OK] Latency: {result['latency_seconds']}s | Strategies: {result['strategies']}")
            print(f"    Run ID: {result['run_id']} | Memory Loaded: {result['memory_loaded']}")
            print(f"    Reranked Docs: {result['reranked_count']} | Retries: {result['retry_count']}")
            print(f"    Reflection: is_supported={result['reflection_is_supported']}, score={result['reflection_overall_score']}, should_retry={result['reflection_should_retry']}")
            print(f"    Confidence Score: {result['confidence_score']} ({result['confidence_level'].upper()})")
            print(f"    Answer: {result['answer_preview']!r}...")
        print("-" * 70)

    print("=" * 70)
    if all_successful:
        print("[OK] All benchmark queries completed successfully with zero unhandled exceptions!")
    else:
        print("[WARN] Some queries encountered errors.")
    print(f"Check LangSmith dashboard for full trace details:\n  https://smith.langchain.com/projects")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())

"""
Frontend logic tests — Python-side validation of the SSE contract
between the backend (Module 11's stream.py) and what the frontend
expects to consume.

Tests ensure:
  1. Every node the frontend's _SURFACED_NODES set listens for actually
     exists in the compiled graph
  2. The SSE event shape emitted by stream.py matches what frontend JS
     destructures (node, label, data fields per node)
  3. The 'final' event contains every field the frontend reads

These run without a browser or Node.js — they validate the contract
in Python against the actual stream.py and build_graph.py code.
"""
import sys

import pytest

sys.path.append(".")


FRONTEND_SURFACED_NODES = {
    "memory_load", "query_understanding", "adaptive_router",
    "retrieve", "document_merge", "rerank", "context_validate",
    "generate", "reflect", "confidence_score", "query_rewrite",
    "finalize", "memory_save",
}

FRONTEND_NODE_LABELS = {
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

FINAL_EVENT_REQUIRED_FIELDS = {
    "event", "answer", "citations", "confidence",
    "strategies_used", "retry_count", "conversation_id", "thread_id",
    "clarification_needed",
}


def test_all_surfaced_nodes_exist_in_compiled_graph():
    """Every node the frontend listens for must exist in the real graph —
    if a node is renamed or removed, the frontend silently stops receiving
    events for it without this test."""
    from app.graph.build_graph import build_graph
    compiled = build_graph().compile()
    graph_nodes = set(compiled.get_graph().nodes.keys()) - {"__start__", "__end__"}
    missing = FRONTEND_SURFACED_NODES - graph_nodes
    assert not missing, f"Frontend listens for nodes not in graph: {missing}"


def test_all_graph_nodes_are_surfaced_or_intentionally_excluded():
    """Inverse check: every graph node should either be in SURFACED_NODES
    or we should be intentionally choosing to hide it. Right now all
    nodes should be surfaced."""
    from app.graph.build_graph import build_graph
    compiled = build_graph().compile()
    graph_nodes = set(compiled.get_graph().nodes.keys()) - {"__start__", "__end__"}
    unsurfaced = graph_nodes - FRONTEND_SURFACED_NODES
    assert not unsurfaced, (
        f"Graph nodes not surfaced in frontend: {unsurfaced}. "
        "Add to FRONTEND_SURFACED_NODES in stream.py and frontend/index.html "
        "or document why they're intentionally hidden."
    )


def test_all_surfaced_nodes_have_labels():
    """Every surfaced node must have a human-readable label for the
    frontend's progress panel — a missing label shows the raw node ID."""
    missing_labels = FRONTEND_SURFACED_NODES - set(FRONTEND_NODE_LABELS.keys())
    assert not missing_labels, f"Nodes missing frontend labels: {missing_labels}"


def test_stream_node_labels_match_frontend():
    """Backend stream.py's _NODE_LABELS must match the frontend's
    expected labels — these are what the user sees in the progress panel."""
    from app.api.stream import _NODE_LABELS as backend_labels
    for node, expected_label in FRONTEND_NODE_LABELS.items():
        assert node in backend_labels, f"Node '{node}' missing from stream._NODE_LABELS"
        assert backend_labels[node] == expected_label, (
            f"Label mismatch for '{node}': "
            f"backend='{backend_labels[node]}' frontend='{expected_label}'"
        )


def test_final_event_fields_are_produced_by_stream_endpoint():
    """Verify stream.py's final event payload contains every field the
    frontend destructures, checked by inspecting the source."""
    import inspect
    from app.api import stream
    source = inspect.getsource(stream._event_generator)

    for field in FINAL_EVENT_REQUIRED_FIELDS:
        assert f'"{field}"' in source or f"'{field}'" in source, (
            f"Final event field '{field}' not found in stream._event_generator source. "
            "The frontend will silently receive undefined for this field."
        )


def test_sse_data_prefix_format():
    """SSE lines must start with 'data: ' — verify the _sse helper
    produces the correct format."""
    from app.api.stream import _sse
    import json
    payload = {"event": "node_complete", "node": "generate"}
    result = _sse(payload)
    assert result.startswith("data: ")
    assert result.endswith("\n\n")
    parsed = json.loads(result[6:])
    assert parsed["event"] == "node_complete"


def test_node_complete_event_includes_label():
    """The frontend reads event.label to display in the progress panel.
    Verify stream.py always includes it for surfaced nodes."""
    import inspect
    from app.api import stream
    source = inspect.getsource(stream._event_generator)
    assert '"label"' in source or "'label'" in source, (
        "stream._event_generator doesn't emit a 'label' field in node_complete events. "
        "Frontend progress panel will show raw node IDs."
    )

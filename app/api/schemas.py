"""
API boundary schemas.
Separate from app/graph/state.py (internal) and app/schemas/* (LLM output
validation) — these are the HTTP contract, versioned independently of the
graph's internal state shape. A change to GraphState's internal fields
doesn't have to change the API response shape, and vice versa.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import uuid


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    conversation_id: Optional[str] = Field(
        default=None,
        description="Persistent ID for multi-turn conversations. "
                    "Generated server-side if omitted.",
    )
    thread_id: Optional[str] = Field(
        default=None,
        description="LangGraph thread ID for checkpointer resumability. "
                    "Defaults to conversation_id if omitted.",
    )

    def resolved_conversation_id(self) -> str:
        return self.conversation_id or str(uuid.uuid4())

    def resolved_thread_id(self, conversation_id: str) -> str:
        return self.thread_id or conversation_id


class ConfidenceDetail(BaseModel):
    confidence_score: float
    hallucination_risk: float
    retrieval_quality: float
    reflection_score: float
    citation_quality: float
    num_sources: int
    confidence_level: str
    reason: str


class QueryResponse(BaseModel):
    answer: str
    citations: List[str]
    confidence: ConfidenceDetail
    strategies_used: List[str]
    retry_count: int
    conversation_id: str
    thread_id: str
    clarification_needed: bool = False
    clarification_question: Optional[str] = None
    memory_context: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StreamEvent(BaseModel):
    """Single SSE payload — one per completed node during streaming."""
    event: str   # "node_complete" | "final" | "error"
    node: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    graph_compiled: bool
    checkpointer_backend: str
    store_backend: str


class GraphStructureResponse(BaseModel):
    nodes: List[str]
    edges: List[Dict[str, str]]

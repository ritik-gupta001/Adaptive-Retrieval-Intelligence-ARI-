class ARIError(Exception):
    """Base class for all ARI platform errors."""


class ConfigError(ARIError):
    """Raised when required configuration is missing or invalid."""


class LLMCallError(ARIError):
    """Raised when an LLM call fails or returns unparseable output after retries."""


class RetrievalError(ARIError):
    """Raised when a retrieval strategy fails (e.g. vector store unreachable)."""


class RerankError(ARIError):
    """Raised when reranking fails (model load failure, etc.) — nodes should
    catch this internally and degrade gracefully, not let it reach the API."""


class ValidationFailedError(ARIError):
    """Raised when context validation hard-fails and there's no retry budget left."""


class GraphExecutionError(ARIError):
    """Wraps any node-level error with the run_id + node name for the API layer."""

    def __init__(self, message: str, node: str, run_id: str | None = None):
        self.node = node
        self.run_id = run_id
        super().__init__(f"[node={node} run_id={run_id}] {message}")

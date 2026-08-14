"""
Central configuration settings for the Adaptive Retrieval Intelligence (ARI) platform.

Parses environment variables and .env settings using Pydantic BaseSettings, enforcing
type validation, sensible defaults, and provider API key validation.
"""

import json
from functools import lru_cache
from typing import List, Literal, Optional, Union

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Priority (last wins): local .env → Render secret file → actual env vars
    # Render mounts Secret Files at /etc/secrets/<filename> — we support both
    # the local dev path and the Render production path automatically.
    model_config = SettingsConfigDict(
        env_file=(".env", "/etc/secrets/.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def _fallback_provider(self) -> "Settings":
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            if self.openai_api_key:
                self.llm_provider = "openai"
                if self.llm_model == "claude-sonnet-4-6":
                    self.llm_model = "gpt-4o-mini"
            elif self.google_api_key:
                self.llm_provider = "gemini"
                if self.llm_model == "claude-sonnet-4-6":
                    self.llm_model = "gemini-1.5-flash"
            elif self.groq_api_key:
                self.llm_provider = "groq"
                if self.llm_model == "claude-sonnet-4-6":
                    self.llm_model = "mixtral-8x7b-32768"
        return self

    # --- Environment ---
    environment: Literal["dev", "staging", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    api_key: Optional[str] = None

    # --- LLM ---
    llm_provider: Literal["openai", "anthropic", "groq", "gemini"] = "anthropic"
    llm_model: str = "claude-sonnet-4-6"
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    google_api_key: Optional[str] = None

    # --- Vector store ---
    vector_store: Literal["chroma", "faiss", "qdrant"] = "chroma"
    chroma_dir: str = "./data/chroma"
    faiss_index_path: str = "./data/faiss_index"
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    collection_name: str = "ari_default"

    # --- Reranking ---
    reranker_provider: Literal["bge", "cohere", "none"] = "bge"
    cohere_api_key: Optional[str] = None
    bge_model_name: str = "BAAI/bge-reranker-base"

    # --- LLM runtime tuning ---
    llm_client_timeout: float = 12.0           # per-client HTTP timeout
    llm_client_max_retries: int = 1            # HTTP retries before switching provider
    llm_classification_timeout: float = 10.0  # fast nodes (router, query_understanding)

    # --- Fallback model per provider (used when not the primary provider) ---
    openai_fallback_model: str = "gpt-4o-mini"
    groq_fallback_model: str = "llama-3.3-70b-versatile"
    anthropic_fallback_model: str = "claude-3-5-sonnet-20241022"
    gemini_fallback_model: str = "gemini-1.5-flash"

    # --- Token budgets per node ---
    max_tokens_classification: int = 256   # query_understanding, router
    max_tokens_validation: int = 300       # context_validate
    max_tokens_reflection: int = 400       # reflect
    max_tokens_generation: int = 1500      # final_answer generation

    # --- Document truncation per node ---
    max_doc_chars_generation: int = 2500   # chars per doc in generate context
    max_doc_chars_reflection: int = 1000   # chars per doc in reflection context
    max_doc_chars_validation: int = 600    # chars per doc in validation context

    # --- Confidence scoring ---
    confidence_high_threshold: float = 0.75
    confidence_medium_threshold: float = 0.5
    web_search_confidence_cap: float = 0.72  # max confidence for web-only answers
    confidence_weight_reflection: float = 0.35
    confidence_weight_retrieval: float = 0.30
    confidence_weight_hallucination: float = 0.25
    confidence_weight_citation: float = 0.10

    # --- Memory ---
    memory_history_turns: int = 5           # how many past turns to load as context
    memory_answer_preview_chars: int = 300  # chars of each past answer to include

    # --- File upload processing ---
    upload_chunk_size: int = 500
    upload_chunk_overlap: int = 50
    uploaded_file_extensions: Union[List[str], str] = Field(
        default_factory=lambda: [".pdf", ".txt", ".doc", ".docx", ".csv", ".md"]
    )

    # --- Graph execution ---
    graph_recursion_limit: int = 50

    # --- Web search ---
    tavily_api_key: Optional[str] = None

    # --- Graph RAG (optional) ---
    graph_rag_enabled: bool = False
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: Optional[str] = None

    # --- Retrieval tuning ---
    top_k: int = 5
    bm25_weight: float = 0.4
    vector_weight: float = 0.6
    multi_query_n: int = 4
    document_reference_keywords: Union[List[str], str] = Field(default_factory=list)

    # --- Reflection / confidence loop ---
    confidence_threshold: float = 0.7
    hallucination_risk_threshold: float = 0.3
    max_retries: int = 2

    # --- Memory ---
    checkpointer_backend: Literal["memory", "postgres", "sqlite"] = "memory"
    checkpointer_conn_str: Optional[str] = None
    store_backend: Literal["memory", "postgres"] = "memory"

    # --- Observability ---
    langsmith_tracing_enabled: bool = False
    langsmith_api_key: Optional[str] = None
    langsmith_project: str = "ari-platform"

    # --- Evaluation ---
    eval_framework: Literal["ragas", "deepeval", "both"] = "both"

    @field_validator("uploaded_file_extensions", "document_reference_keywords", mode="after")
    @classmethod
    def _parse_list_fields(cls, v: Union[List[str], str]) -> List[str]:
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("[") and v.endswith("]"):
                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, list):
                        return [str(x).strip() for x in parsed]
                except Exception:
                    pass
            return [x.strip() for x in v.split(",") if x.strip()]
        elif isinstance(v, list):
            return [str(x).strip() for x in v]
        return v

    @field_validator("bm25_weight", "vector_weight")
    @classmethod
    def _weight_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("retrieval fusion weights must be between 0 and 1")
        return v

    @property
    def enabled_strategies(self) -> List[str]:
        """Computed, not stored — recalculates from current flags so a runtime
        env change (e.g. in tests) is always reflected correctly."""
        strategies = ["vector_search", "hybrid_search", "multi_query_retrieval"]
        if self.tavily_api_key:
            strategies.append("web_search")
        if self.graph_rag_enabled:
            strategies.append("graph_rag")
        return strategies

    def active_llm_key(self) -> str:
        """Resolve which API key matters for the configured provider, and
        fail loudly (not at first LLM call 3 nodes deep) if it's missing."""
        key_map = {
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
            "groq": self.groq_api_key,
            "gemini": self.google_api_key,
        }
        key = key_map.get(self.llm_provider)
        if not key:
            raise ValueError(
                f"llm_provider is '{self.llm_provider}' but its API key is not set. "
                f"Check your .env."
            )
        return key


@lru_cache
def get_settings() -> Settings:
    """Cached singleton — settings are read constantly across nodes; no
    reason to re-parse .env on every call."""
    return Settings()


settings = get_settings()

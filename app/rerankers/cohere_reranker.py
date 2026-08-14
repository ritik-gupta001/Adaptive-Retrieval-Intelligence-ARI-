"""
Cohere Rerank — hosted API, no local model weights to load, lower cold-
start latency than BGE but requires an API key and network access.
"""
import asyncio
from typing import List

from app.config.settings import settings
from app.core.exceptions import RerankError
from app.core.logging import get_logger
from app.graph.state import Document
from app.rerankers.base import BaseReranker

logger = get_logger(__name__)


class CohereReranker(BaseReranker):
    name = "cohere"

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not settings.cohere_api_key:
            raise RerankError("cohere reranker selected but COHERE_API_KEY is not configured")
        try:
            import cohere
        except ImportError as exc:
            raise RerankError(f"cohere SDK not installed: {exc}") from exc

        self._client = cohere.Client(api_key=settings.cohere_api_key)
        return self._client

    def _rerank_sync(self, query: str, docs: List[Document], top_n: int):
        client = self._get_client()
        documents = [d.get("content", "") for d in docs]
        return client.rerank(
            query=query, documents=documents, top_n=min(top_n, len(documents)), model="rerank-english-v3.0"
        )

    async def rerank(self, query: str, docs: List[Document], top_n: int) -> List[Document]:
        if not docs:
            return []
        try:
            response = await asyncio.to_thread(self._rerank_sync, query, docs, top_n)
        except RerankError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("cohere_rerank_failed", extra={"error": str(exc)})
            raise RerankError(f"Cohere rerank failed: {exc}") from exc

        reranked = []
        for result in response.results:
            doc = dict(docs[result.index])
            doc["score"] = float(result.relevance_score)
            reranked.append(doc)
        return reranked

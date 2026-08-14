"""
BGE Reranker — local cross-encoder (sentence-transformers), no external API
call. Good default since it requires no extra API key and runs offline,
but is heavier at startup (model download/load) than the Cohere option.

Lazy model loading: the CrossEncoder is only instantiated on first use, not
at import time — importing this module (e.g. transitively via the registry)
shouldn't trigger a multi-hundred-MB model download in a deployment that's
configured to use Cohere instead.
"""
import asyncio
from typing import List

from app.config.settings import settings
from app.core.exceptions import RerankError
from app.core.logging import get_logger
from app.graph.state import Document
from app.rerankers.base import BaseReranker

logger = get_logger(__name__)


class BGEReranker(BaseReranker):
    name = "bge"

    def __init__(self):
        self._model = None

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RerankError(f"sentence-transformers not installed: {exc}") from exc

        try:
            self._model = CrossEncoder(settings.bge_model_name)
        except Exception as exc:  # noqa: BLE001
            raise RerankError(f"Failed to load BGE model '{settings.bge_model_name}': {exc}") from exc
        return self._model

    def _score_sync(self, query: str, docs: List[Document]) -> List[float]:
        model = self._get_model()
        pairs = [(query, d.get("content", "")) for d in docs]
        scores = model.predict(pairs)
        return [float(s) for s in scores]

    async def rerank(self, query: str, docs: List[Document], top_n: int) -> List[Document]:
        if not docs:
            return []
        try:
            scores = await asyncio.to_thread(self._score_sync, query, docs)
        except RerankError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("bge_rerank_failed", extra={"error": str(exc)})
            raise RerankError(f"BGE rerank failed: {exc}") from exc

        scored = []
        for doc, score in zip(docs, scores):
            d = dict(doc)
            d["score"] = score
            scored.append(d)

        scored.sort(key=lambda d: d["score"], reverse=True)
        return scored[:top_n]

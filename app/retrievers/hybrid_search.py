
import asyncio
import json
from pathlib import Path
from typing import Dict, List

from app.config.settings import settings
from app.core.exceptions import RetrievalError
from app.core.logging import get_logger
from app.graph.state import Document
from app.retrievers.base import BaseRetriever, BM25_INDEX_PATH
from app.retrievers.vector_search import VectorSearchRetriever
from app.utils.query_intent import is_document_reference_query

logger = get_logger(__name__)

RRF_K = 60



_GLOBAL_BM25_INDEX = None


class BM25Index:
    """Lazy-loaded, process-local singleton BM25 index used internally by HybridSearchRetriever."""

    def __init__(self):
        self._bm25 = None
        self._corpus: List[Dict] = []

    def _load(self):
        if self._bm25 is not None:
            return
        if not BM25_INDEX_PATH.exists():
            logger.warning(f"BM25 corpus not found at {BM25_INDEX_PATH}")
            self._corpus = []
            return
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank_bm25 not installed")
            self._bm25 = None
            return

        with open(BM25_INDEX_PATH, encoding="utf-8") as f:
            self._corpus = json.load(f)
        tokenized = [doc["content"].lower().split() for doc in self._corpus]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int) -> List[Document]:
        self._load()
        if not self._bm25 or not self._corpus:
            return []

        scores = list(self._bm25.get_scores(query.lower().split()))
        is_about_file = is_document_reference_query(query)

        # Boost score if the doc is from an uploaded source and query is about files
        for i, doc in enumerate(self._corpus):
            source = doc.get("source", "")
            if is_about_file and source and not source.startswith("data"):
                scores[i] += 2.0

        ranked = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]
        max_score = max(scores) if len(scores) and max(scores) > 0 else 1.0
        return [
            Document(
                content=self._corpus[i]["content"],
                source=self._corpus[i].get("source", "unknown"),
                score=float(scores[i] / max_score),  # normalize to 0-1
                metadata=self._corpus[i].get("metadata", {}),
            )
            for i in ranked
        ]


def get_global_bm25_index() -> BM25Index:
    global _GLOBAL_BM25_INDEX
    if _GLOBAL_BM25_INDEX is None:
        _GLOBAL_BM25_INDEX = BM25Index()
        _GLOBAL_BM25_INDEX._load()
    return _GLOBAL_BM25_INDEX


class HybridSearchRetriever(BaseRetriever):
    name = "hybrid_search"

    def __init__(self):
        self._vector = VectorSearchRetriever()
        self._bm25 = get_global_bm25_index()

    @staticmethod
    def _rrf_fuse(
        ranked_lists: List[List[Document]], k: int, weights: List[float]
    ) -> List[Document]:
        """ranked_lists[i] is already sorted best-first; weights[i] scales
        that list's contribution (settings.vector_weight / bm25_weight)."""
        scores: Dict[str, float] = {}
        doc_lookup: Dict[str, Document] = {}

        for ranked_list, weight in zip(ranked_lists, weights):
            for rank, doc in enumerate(ranked_list):
                key = doc["content"].strip().lower()
                rrf_contribution = weight * (1.0 / (RRF_K + rank + 1))
                scores[key] = scores.get(key, 0.0) + rrf_contribution
                if key not in doc_lookup:
                    doc_lookup[key] = doc

        fused = []
        for key, score in scores.items():
            doc = dict(doc_lookup[key])
            doc["score"] = score
            fused.append(doc)

        fused.sort(key=lambda d: d["score"], reverse=True)
        return fused[:k]

    async def retrieve(self, query: str, k: int) -> List[Document]:
        try:
            vector_docs = await self._vector.retrieve(query, k=k * 2)
        except RetrievalError as exc:
            logger.warning("hybrid_vector_leg_failed", extra={"error": str(exc)})
            vector_docs = []

        try:
            bm25_docs = await asyncio.to_thread(self._bm25.search, query, k * 2)
        except RetrievalError as exc:
            logger.warning("hybrid_bm25_leg_failed", extra={"error": str(exc)})
            bm25_docs = []

        if not vector_docs and not bm25_docs:
            raise RetrievalError("hybrid_search: both vector and BM25 legs failed")

        fused = self._rrf_fuse(
            [vector_docs, bm25_docs], k=k, weights=[settings.vector_weight, settings.bm25_weight]
        )
        return self._tag(fused)

"""
Strategy 1: Vector Search — dense embedding similarity search using ChromaDB.
"""

import asyncio
from typing import List

from app.config.settings import settings
from app.core.exceptions import RetrievalError
from app.core.logging import get_logger
from app.graph.state import Document
from app.retrievers.base import BaseRetriever, BM25_INDEX_PATH, get_uploaded_sources, parse_chromadb_results

from app.utils.dedup import dedupe_documents
from app.utils.query_intent import is_document_reference_query

logger = get_logger(__name__)


class VectorSearchRetriever(BaseRetriever):
    name = "vector_search"

    def __init__(self):
        self._collection = None

    def _get_collection(self):
        if self._collection is not None:
            return self._collection

        if settings.vector_store != "chroma":
            raise RetrievalError(
                f"vector_store='{settings.vector_store}' is configured but only "
                f"'chroma' is implemented in this module — FAISS/Qdrant are "
                f"config slots for a future iteration."
            )

        try:
            import chromadb
        except ImportError as exc:
            raise RetrievalError(f"chromadb not installed: {exc}") from exc

        from chromadb.config import Settings as ChromaSettings
        client = chromadb.PersistentClient(
            path=settings.chroma_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = client.get_or_create_collection(
            name=settings.collection_name, metadata={"hnsw:space": "cosine"}
        )
        space = (self._collection.metadata or {}).get("hnsw:space")
        if space and space != "cosine":
            logger.warning(
                "chroma_collection_space_mismatch",
                extra={"expected": "cosine", "actual": space, "collection": settings.collection_name},
            )
        return self._collection

    async def retrieve(self, query: str, k: int) -> List[Document]:
        uploaded_sources = get_uploaded_sources(BM25_INDEX_PATH)
        is_about_file = is_document_reference_query(query)

        try:
            collection = self._get_collection()
            results = await asyncio.to_thread(
                collection.query, query_texts=[query], n_results=k
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("vector_search_failed", extra={"query": query, "error": str(exc)})
            raise RetrievalError(f"vector_search failed: {exc}") from exc

        uploaded_results = None
        if uploaded_sources and is_about_file:
            try:
                where_filter = {"source": uploaded_sources[0]} if len(uploaded_sources) == 1 else {"source": {"$in": uploaded_sources}}
                uploaded_results = await asyncio.to_thread(
                    collection.query, query_texts=[query], n_results=k, where=where_filter
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("uploaded_vector_search_failed", extra={"query": query, "error": str(exc)})

        base_docs = parse_chromadb_results(results)
        uploaded_docs = parse_chromadb_results(uploaded_results)

        # Merge and deduplicate by content
        deduped = dedupe_documents(uploaded_docs + base_docs)
        return self._tag(deduped[:k])

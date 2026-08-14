"""
Strategy 3: Multi-Query Retrieval.
Generates diverse sub-queries via LLM, retrieves documents for each concurrently,
merges, and deduplicates the results.
"""

import asyncio
from typing import List

from pydantic import ValidationError

from app.config.settings import settings
from app.core.exceptions import LLMCallError, RetrievalError
from app.core.llm import call_llm_json
from app.core.logging import get_logger
from app.graph.state import Document
from app.retrievers.base import BaseRetriever
from app.retrievers.vector_search import VectorSearchRetriever
from app.schemas.multi_query import MultiQueryOutput
from app.utils.dedup import dedupe_documents

from app.retrievers.hybrid_search import HybridSearchRetriever

logger = get_logger(__name__)


class MultiQueryRetriever(BaseRetriever):
    name = "multi_query_retrieval"

    def __init__(self):
        self._base_retriever = HybridSearchRetriever()

    async def _generate_queries(self, question: str) -> List[str]:
        try:
            raw = await call_llm_json(
                "multi_query_generation", question=question, n=settings.multi_query_n
            )
            validated = MultiQueryOutput.model_validate(raw)
            queries = validated.queries[: settings.multi_query_n]
            return queries or [question]
        except (LLMCallError, ValidationError) as exc:
            logger.warning(
                "multi_query_generation_failed_using_original_question",
                extra={"question": question, "error": str(exc)},
            )
            return [question]

    async def retrieve(self, query: str, k: int) -> List[Document]:
        sub_queries = await self._generate_queries(query)
        per_query_k = max(2, k // max(1, len(sub_queries)) + 1)

        # Run all sub-query retrievals concurrently (not sequentially).
        # This is what the docstring and design always promised — sub-queries are
        # independent so there is no reason to await each one before starting the next.
        gather_results = await asyncio.gather(
            *[self._base_retriever.retrieve(sq, per_query_k) for sq in sub_queries],
            return_exceptions=True,
        )

        all_docs: List[Document] = []
        succeeded = 0
        for sub_query, result in zip(sub_queries, gather_results):
            if isinstance(result, Exception):
                logger.warning(
                    "multi_query_sub_query_failed",
                    extra={"sub_query": sub_query, "error": str(result)},
                )
                continue
            all_docs.extend(result)
            succeeded += 1

        if succeeded == 0:
            raise RetrievalError("multi_query_retrieval: every sub-query failed")

        deduped = dedupe_documents(all_docs)[:k]
        return self._tag(deduped)


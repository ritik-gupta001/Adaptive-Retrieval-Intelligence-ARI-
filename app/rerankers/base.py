"""
Common interface for rerank providers — mirrors retrievers/base.py's
pattern so the provider switch in nodes/rerank.py stays a simple dict
lookup, not a chain of if/elif special cases.
"""
from abc import ABC, abstractmethod
from typing import List

from app.graph.state import Document


class BaseReranker(ABC):
    name: str = "base"

    @abstractmethod
    async def rerank(self, query: str, docs: List[Document], top_n: int) -> List[Document]:
        """Return up to top_n docs from `docs`, re-scored and re-sorted by
        the reranker's own judgment of (query, document) relevance."""
        raise NotImplementedError

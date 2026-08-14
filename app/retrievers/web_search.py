"""
Strategy 4: Web Search — Tavily API integration.
Retrieves external web search results for queries requiring real-time information.
"""

import asyncio
from typing import List

from app.config.settings import settings
from app.core.exceptions import RetrievalError
from app.core.logging import get_logger
from app.graph.state import Document
from app.retrievers.base import BaseRetriever

logger = get_logger(__name__)


class WebSearchRetriever(BaseRetriever):
    name = "web_search"

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not settings.tavily_api_key:
            raise RetrievalError(
                "web_search strategy selected but TAVILY_API_KEY is not configured"
            )
        from tavily import TavilyClient
        self._client = TavilyClient(api_key=settings.tavily_api_key)
        return self._client

    async def retrieve(self, query: str, k: int) -> List[Document]:
        docs: List[Document] = []
        try:
            client = self._get_client()
            response = await asyncio.to_thread(
                client.search, query=query, max_results=k, search_depth="advanced"
            )
            for i, result in enumerate(response.get("results", [])[:k]):
                score = result.get("score")
                if score is None:
                    score = max(0.1, 1.0 - (i * 0.1))
                docs.append(
                    Document(
                        content=result.get("content", ""),
                        source=result.get("url", "unknown"),
                        score=float(score),
                        metadata={"title": result.get("title", "")},
                    )
                )
            if docs:
                return self._tag(docs)
        except RetrievalError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("tavily_web_search_failed_falling_back_to_ddg", extra={"query": query, "error": str(exc)})

        # Free fallback web search via DuckDuckGo if Tavily fails or is absent
        try:
            from langchain_community.tools import DuckDuckGoSearchRun
            search_tool = DuckDuckGoSearchRun()
            snippet = await asyncio.to_thread(search_tool.run, query)
            if snippet and snippet.strip():
                docs.append(
                    Document(
                        content=snippet,
                        source="web_search_ddg",
                        score=0.85,
                        metadata={"title": f"Web Search: {query}"},
                    )
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("ddg_web_search_failed", extra={"query": query, "error": str(exc)})

        if not docs:
            raise RetrievalError(f"web_search failed to retrieve content for query: '{query}'")

        return self._tag(docs)

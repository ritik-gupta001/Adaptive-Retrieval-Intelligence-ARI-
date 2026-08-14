"""
Strategy 5: Graph RAG — Neo4j entity and relationship traversal.
Retrieves connected entity nodes and multi-hop relationship graphs.
"""
import asyncio
from typing import List

from app.config.settings import settings
from app.core.exceptions import RetrievalError
from app.core.logging import get_logger
from app.graph.state import Document
from app.retrievers.base import BaseRetriever

logger = get_logger(__name__)


class GraphRAGRetriever(BaseRetriever):
    name = "graph_rag"

    def __init__(self):
        self._driver = None

    def _get_driver(self):
        if not settings.graph_rag_enabled:
            raise RetrievalError(
                "graph_rag strategy invoked but GRAPH_RAG_ENABLED is false. "
                "Check configuration settings before invoking Graph RAG."
            )

        if self._driver is not None:
            return self._driver

        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
        )
        return self._driver

    def _query_sync(self, query: str, k: int) -> List[Document]:
        driver = self._get_driver()
        docs: List[Document] = []
        with driver.session() as session:
            # Seed entities via fulltext match
            seeds = session.run(
                """
                CALL db.index.fulltext.queryNodes('entityIndex', $query)
                YIELD node, score
                RETURN node.name AS name, node.description AS description, score
                ORDER BY score DESC
                LIMIT $limit
                """,
                query=query,
                limit=max(1, k // 2),
            ).data()

            for seed in seeds:
                docs.append(
                    Document(
                        content=f"{seed['name']}: {seed.get('description', '')}",
                        source=f"neo4j:entity:{seed['name']}",
                        score=float(seed.get("score", 0.5)),
                        metadata={"type": "seed_entity"},
                    )
                )

                # One-hop expansion for relational context
                related = session.run(
                    """
                    MATCH (a:Entity {name: $name})-[:RELATED_TO]-(b:Entity)
                    RETURN b.name AS name, b.description AS description
                    LIMIT $limit
                    """,
                    name=seed["name"],
                    limit=3,
                ).data()
                for rel in related:
                    docs.append(
                        Document(
                            content=f"{rel['name']} (related to {seed['name']}): "
                                    f"{rel.get('description', '')}",
                            source=f"neo4j:related:{rel['name']}",
                            score=float(seed.get("score", 0.5)) * 0.8,  # one hop = slightly lower confidence
                            metadata={"type": "related_entity", "related_to": seed["name"]},
                        )
                    )
        return docs[:k]

    async def retrieve(self, query: str, k: int) -> List[Document]:
        try:
            docs = await asyncio.to_thread(self._query_sync, query, k)
        except RetrievalError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("graph_rag_failed", extra={"query": query, "error": str(exc)})
            raise RetrievalError(f"graph_rag failed: {exc}") from exc

        return self._tag(docs)

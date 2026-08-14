"""
Ingestion script for Neo4j Graph RAG index.

WHY this script exists:
GraphRAGRetriever (app/retrievers/graph_rag.py) performs fulltext entity matching
via Neo4j's 'entityIndex' fulltext index and one-hop relational expansion via
[:RELATED_TO] edges. This script creates the fulltext index and populates the graph
with seed entities and relationships.
"""
import argparse
from typing import Dict, List, Tuple

from app.config.settings import settings
from app.core.exceptions import ARIError
from app.core.logging import get_logger

logger = get_logger(__name__)

CREATE_FULLTEXT_INDEX_CYPHER = """
CREATE FULLTEXT INDEX entityIndex IF NOT EXISTS
FOR (n:Entity) ON EACH [n.name, n.description]
"""

SAMPLE_ENTITIES = [
    {"name": "LangGraph", "description": "Framework for building stateful, multi-actor applications with LLMs."},
    {"name": "StateGraph", "description": "Core graph abstraction in LangGraph managing node transitions and state."},
    {"name": "Checkpointer", "description": "Persistence mechanism in LangGraph saving state snapshots per thread."},
    {"name": "Adaptive Router", "description": "Routing node selecting retrieval strategies based on query intent."},
    {"name": "Vector Search", "description": "Dense embedding similarity search strategy using vector databases."},
    {"name": "Hybrid Search", "description": "Combined BM25 and vector retrieval fused with Reciprocal Rank Fusion."},
]

SAMPLE_RELATIONS: List[Tuple[str, str, str]] = [
    ("LangGraph", "USES", "StateGraph"),
    ("StateGraph", "RELATED_TO", "Checkpointer"),
    ("Adaptive Router", "RELATED_TO", "Vector Search"),
    ("Adaptive Router", "RELATED_TO", "Hybrid Search"),
    ("Vector Search", "RELATED_TO", "Hybrid Search"),
]


def create_fulltext_index(session) -> None:
    """Create the required entityIndex fulltext index in Neo4j if missing."""
    session.run(CREATE_FULLTEXT_INDEX_CYPHER)
    logger.info("neo4j_fulltext_index_created", extra={"index_name": "entityIndex"})


def ingest_entities(session, entities: List[Dict[str, str]]) -> None:
    """Ingest/merge entity nodes into Neo4j."""
    for entity in entities:
        session.run(
            """
            MERGE (e:Entity {name: $name})
            SET e.description = $description
            """,
            name=entity["name"],
            description=entity.get("description", ""),
        )
    logger.info("neo4j_entities_ingested", extra={"count": len(entities)})


def ingest_relations(session, relations: List[Tuple[str, str, str]]) -> None:
    """Ingest/merge relational edges into Neo4j."""
    for source_name, rel_type, target_name in relations:
        session.run(
            f"""
            MATCH (a:Entity {{name: $source}})
            MATCH (b:Entity {{name: $target}})
            MERGE (a)-[:{rel_type}]->(b)
            """,
            source=source_name,
            target=target_name,
        )
    logger.info("neo4j_relations_ingested", extra={"count": len(relations)})


def build_graph_index(
    uri: str = settings.neo4j_uri,
    user: str = settings.neo4j_user,
    password: str = settings.neo4j_password or "",
    entities: List[Dict[str, str]] = SAMPLE_ENTITIES,
    relations: List[Tuple[str, str, str]] = SAMPLE_RELATIONS,
) -> None:
    """Connect to Neo4j, build fulltext index, and populate knowledge graph."""
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise ARIError(f"neo4j driver not installed: {exc}") from exc

    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        create_fulltext_index(session)
        ingest_entities(session, entities)
        ingest_relations(session, relations)
    driver.close()
    logger.info("build_graph_index_complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Neo4j fulltext index and populate graph RAG entities.")
    parser.add_argument("--uri", default=settings.neo4j_uri, help="Neo4j URI")
    parser.add_argument("--user", default=settings.neo4j_user, help="Neo4j username")
    parser.add_argument("--password", default=settings.neo4j_password or "", help="Neo4j password")
    args = parser.parse_args()

    build_graph_index(uri=args.uri, user=args.user, password=args.password)


if __name__ == "__main__":
    main()

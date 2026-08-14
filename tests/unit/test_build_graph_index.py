"""
Unit tests for ingestion/build_graph_index.py.

Tests requirement 6:
- Verify fulltext index creation and entity/relation ingestion functions with a mocked Neo4j session.
"""
from unittest.mock import MagicMock
from ingestion.build_graph_index import (
    create_fulltext_index,
    ingest_entities,
    ingest_relations,
)


def test_create_fulltext_index():
    session = MagicMock()
    create_fulltext_index(session)
    session.run.assert_called_once()
    cypher_call = session.run.call_args[0][0]
    assert "entityIndex" in cypher_call


def test_ingest_entities():
    session = MagicMock()
    entities = [{"name": "TestNode", "description": "Test description"}]
    ingest_entities(session, entities)
    assert session.run.call_count == 1


def test_ingest_relations():
    session = MagicMock()
    relations = [("NodeA", "RELATED_TO", "NodeB")]
    ingest_relations(session, relations)
    assert session.run.call_count == 1

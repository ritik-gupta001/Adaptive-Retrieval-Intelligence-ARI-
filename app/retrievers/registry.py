"""
Single dispatch point from router strategy names -> retriever instances.

Lazy construction matters: instantiating GraphRAGRetriever or
WebSearchRetriever at import time would mean every process boot attempts
(or at least prepares for) a Neo4j/Tavily connection even when those
strategies are disabled. Instances are built on first use and cached.

Adding a 6th strategy: implement BaseRetriever, add one line to
_FACTORIES. No changes needed in nodes/retrieve.py.
"""
from typing import Callable, Dict

from app.retrievers.base import BaseRetriever
from app.retrievers.graph_rag import GraphRAGRetriever
from app.retrievers.hybrid_search import HybridSearchRetriever
from app.retrievers.multi_query import MultiQueryRetriever
from app.retrievers.vector_search import VectorSearchRetriever
from app.retrievers.web_search import WebSearchRetriever

_FACTORIES: Dict[str, Callable[[], BaseRetriever]] = {
    "vector_search": VectorSearchRetriever,
    "hybrid_search": HybridSearchRetriever,
    "multi_query_retrieval": MultiQueryRetriever,
    "web_search": WebSearchRetriever,
    "graph_rag": GraphRAGRetriever,
}

_instances: Dict[str, BaseRetriever] = {}


def get_retriever(strategy_name: str) -> BaseRetriever:
    if strategy_name not in _FACTORIES:
        raise KeyError(
            f"Unknown retrieval strategy '{strategy_name}'. "
            f"Available: {list(_FACTORIES.keys())}"
        )
    if strategy_name not in _instances:
        _instances[strategy_name] = _FACTORIES[strategy_name]()
    return _instances[strategy_name]


def available_strategy_names() -> list:
    return list(_FACTORIES.keys())


def clear_retriever_caches():
    _instances.clear()

"""
Common interface every retrieval strategy implements. Also contains shared helper
utilities for Chroma result parsing, source filtering, and document tagging across strategies.
"""
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.graph.state import Document

BM25_INDEX_PATH = Path("./data/bm25_corpus.json")


def parse_chromadb_results(results: Optional[Dict[str, Any]]) -> List[Document]:
    """
    Parse a ChromaDB query result dictionary into a list of Document dicts.
    Assumes cosine distance where cosine similarity score = max(0.0, 1.0 - distance).
    Requires collection metadata {"hnsw:space": "cosine"}.

    Args:
        results: Dictionary returned by ChromaDB collection.query().

    Returns:
        List[Document]: Parsed Document objects with normalized 0-1 similarity scores.
    """
    if not results:
        return []
    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0] or [{}] * len(ids)

    parsed_docs = []
    for i, content in enumerate(documents):
        distance = distances[i] if i < len(distances) else 1.0
        score = max(0.0, 1.0 - distance)
        meta = metadatas[i] if i < len(metadatas) else {}
        source_val = meta.get("source", ids[i] if i < len(ids) else "unknown")
        parsed_docs.append(
            Document(
                content=content,
                source=source_val,
                score=score,
                metadata=meta,
            )
        )
    return parsed_docs


def get_uploaded_sources(index_path: Path = BM25_INDEX_PATH) -> List[str]:
    """
    Extract non-default uploaded document sources from the BM25 index corpus.

    Args:
        index_path: Path to the bm25_corpus.json index file (defaults to BM25_INDEX_PATH).

    Returns:
        List[str]: Unique source filenames representing user-uploaded files.
    """
    if not index_path.exists():
        return []
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        return list({
            doc.get("source") for doc in corpus
            if doc.get("source") and not doc.get("source").startswith("data")
        })
    except Exception:
        return []


class BaseRetriever(ABC):
    name: str = "base"

    @abstractmethod
    async def retrieve(self, query: str, k: int) -> List[Document]:
        """Return up to k Document dicts. Must tag each doc's 'strategy'
        field with self.name so document_merge / observability can trace
        which retriever produced which result."""
        raise NotImplementedError

    def _tag(self, docs: List[Document]) -> List[Document]:
        for d in docs:
            d["strategy"] = self.name
        return docs

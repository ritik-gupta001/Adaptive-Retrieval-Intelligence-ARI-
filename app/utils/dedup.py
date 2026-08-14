"""
Shared dedup logic used by document_merge_node (across-strategy dedup) and
by multi_query retriever (across-sub-query dedup within a single strategy).

Two passes:
  1. Exact dedup — normalized (lowercased, whitespace-collapsed) content hash.
     Cheap, O(n), catches the common case (same chunk returned by two
     strategies).
  2. Fuzzy dedup — SequenceMatcher ratio above a threshold. Catches
     near-duplicates (e.g. the same chunk with trailing punctuation
     differences, or two overlapping chunks from adjacent text windows).
     O(n^2) so only run after exact dedup has already shrunk the pool.

When two documents are judged duplicates, the higher-scored one is kept —
this matters because the same chunk surfaced by two strategies should keep
whichever strategy's relevance score was more confident.
"""
import re
from difflib import SequenceMatcher
from typing import List

from app.graph.state import Document


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _exact_dedupe(docs: List[Document]) -> List[Document]:
    seen = {}
    for d in docs:
        key = _normalize(d.get("content", ""))
        existing = seen.get(key)
        if existing is None or d.get("score", 0.0) > existing.get("score", 0.0):
            seen[key] = d
    return list(seen.values())


def _fuzzy_dedupe(docs: List[Document], threshold: float) -> List[Document]:
    # Process highest-scored first so the kept copy is always the best one.
    ordered = sorted(docs, key=lambda d: d.get("score", 0.0), reverse=True)
    kept: List[Document] = []
    for doc in ordered:
        content = _normalize(doc.get("content", ""))
        is_dup = False
        for kept_doc in kept:
            kept_content = _normalize(kept_doc.get("content", ""))
            ratio = SequenceMatcher(None, content, kept_content).ratio()
            if ratio >= threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(doc)
    return kept


def dedupe_documents(docs: List[Document], fuzzy_threshold: float = 0.92) -> List[Document]:
    if not docs:
        return []
    exact_deduped = _exact_dedupe(docs)
    fuzzy_deduped = _fuzzy_dedupe(exact_deduped, fuzzy_threshold)
    return sorted(fuzzy_deduped, key=lambda d: d.get("score", 0.0), reverse=True)

import sys

sys.path.append(".")

from app.utils.dedup import dedupe_documents


def test_empty_input_returns_empty():
    assert dedupe_documents([]) == []


def test_exact_duplicates_removed_keeps_higher_score():
    docs = [
        {"content": "The sky is blue.", "score": 0.5},
        {"content": "The sky is blue.", "score": 0.9},
    ]
    result = dedupe_documents(docs)
    assert len(result) == 1
    assert result[0]["score"] == 0.9


def test_case_and_whitespace_insensitive_exact_match():
    docs = [
        {"content": "  The Sky Is Blue.  ", "score": 0.5},
        {"content": "the sky is blue.", "score": 0.5},
    ]
    assert len(dedupe_documents(docs)) == 1


def test_near_duplicates_removed_above_threshold():
    docs = [
        {"content": "The quick brown fox jumps over the lazy dog.", "score": 0.9},
        {"content": "The quick brown fox jumps over the lazy dog!", "score": 0.5},
    ]
    result = dedupe_documents(docs, fuzzy_threshold=0.9)
    assert len(result) == 1
    assert result[0]["score"] == 0.9  # higher-scored copy kept


def test_distinct_documents_both_kept_and_sorted_desc():
    docs = [
        {"content": "Paris is the capital of France.", "score": 0.4},
        {"content": "Tokyo is the capital of Japan.", "score": 0.95},
    ]
    result = dedupe_documents(docs)
    assert len(result) == 2
    assert result[0]["score"] == 0.95


def test_documents_missing_score_default_to_zero_not_crash():
    docs = [{"content": "no score here"}, {"content": "also no score"}]
    result = dedupe_documents(docs)
    assert len(result) == 2


def test_three_way_near_duplicates_keep_only_best():
    docs = [
        {"content": "LangGraph is a library for building agents.", "score": 0.6},
        {"content": "LangGraph is a library for building agents!", "score": 0.95},
        {"content": "LangGraph is a library for building agents.  ", "score": 0.3},
    ]
    result = dedupe_documents(docs, fuzzy_threshold=0.9)
    assert len(result) == 1
    assert result[0]["score"] == 0.95

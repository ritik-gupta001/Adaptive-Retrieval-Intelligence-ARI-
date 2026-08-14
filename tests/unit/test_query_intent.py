
"""
Unit tests for app.utils.query_intent helper.

Tests requirement 1:
- Empty keyword list (never matches)
- Keyword present
- Keyword absent
- Case-insensitivity
"""
import pytest
from app.utils.query_intent import is_document_reference_query


def test_empty_keyword_list_never_matches():
    assert is_document_reference_query("What is in my uploaded resume?", keywords=[]) is False
    assert is_document_reference_query("Check the PDF document", keywords=[]) is False


def test_keyword_present():
    keywords = ["resume", "pdf", "uploaded"]
    assert is_document_reference_query("Can you summarize the uploaded pdf?", keywords=keywords) is True
    assert is_document_reference_query("Here is my resume", keywords=keywords) is True


def test_keyword_absent():
    keywords = ["resume", "pdf", "uploaded"]
    assert is_document_reference_query("What is the capital of France?", keywords=keywords) is False
    assert is_document_reference_query("How does LangGraph work?", keywords=keywords) is False


def test_case_insensitivity():
    keywords = ["Document", "PDF", "RESUME"]
    assert is_document_reference_query("check this document please", keywords=keywords) is True
    assert is_document_reference_query("here is a pdf file", keywords=keywords) is True
    assert is_document_reference_query("MY RESUME IS ATTACHED", keywords=keywords) is True


def test_empty_or_none_question():
    keywords = ["pdf", "doc"]
    assert is_document_reference_query("", keywords=keywords) is False

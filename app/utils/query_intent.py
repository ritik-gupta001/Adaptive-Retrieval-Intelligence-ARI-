"""
Query intent utility functions.

Centralizes document-reference query detection using settings.document_reference_keywords
and common document keywords (pdf, document, file, uploaded, resume, etc.).
"""
from typing import List, Optional

from app.config.settings import settings

DEFAULT_DOCUMENT_KEYWORDS = [
    "pdf", "document", "uploaded", "file", "resume", "paper", "txt", "report", "attachment", "upload"
]


def is_document_reference_query(
    question: str,
    keywords: Optional[List[str]] = None,
) -> bool:
    """
    Determine if `question` references an uploaded document/file based on configured keywords.

    Args:
        question: The user input query string.
        keywords: Optional explicit list of keywords to check. If None, defaults to
                  `settings.document_reference_keywords` merged with `DEFAULT_DOCUMENT_KEYWORDS`.

    Returns:
        bool: True if at least one keyword appears in question (case-insensitive); False otherwise.
    """
    if not question:
        return False

    if keywords is None:
        keywords = list(set(settings.document_reference_keywords + DEFAULT_DOCUMENT_KEYWORDS))

    question_lower = question.lower()
    return any(kw.lower() in question_lower for kw in keywords if kw.strip())

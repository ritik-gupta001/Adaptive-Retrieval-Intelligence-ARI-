"""
Security Guardrails Module for ARI Platform.
Validates input queries for prompt injection, jailbreak attempts, and harmful payloads.
Sanitizes output answers to prevent key leakage or unauthorized instructions.
"""
import re
from typing import Tuple

from app.core.logging import get_logger

logger = get_logger(__name__)

PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|above)\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?previous\s+(rules|instructions)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+in\s+(dan|developer)\s+mode", re.IGNORECASE),
    re.compile(r"system\s+prompt\s+(override|leak|reveal)", re.IGNORECASE),
    re.compile(r"bypass\s+(all\s+)?guardrails", re.IGNORECASE),
    re.compile(r"forget\s+(all\s+)?(previous|prior)\s+(context|instructions)", re.IGNORECASE),
    re.compile(r"reveal\s+(your|the)\s+(system|hidden)\s+(prompt|instructions)", re.IGNORECASE),
    re.compile(r"print\s+(your|the)\s+initial\s+prompt", re.IGNORECASE),
]

SENSITIVE_KEY_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9]{20,}", re.IGNORECASE),
    re.compile(r"tvly-[a-zA-Z0-9]{20,}", re.IGNORECASE),
    re.compile(r"gsk_[a-zA-Z0-9]{20,}", re.IGNORECASE),
]


def validate_input_security(question: str) -> Tuple[bool, str]:
    """Check input question for security risks or prompt injections."""
    for pattern in PROMPT_INJECTION_PATTERNS:
        if pattern.search(question):
            logger.warning("prompt_injection_detected", extra={"question": question})
            return False, "Prompt injection attempt detected. Request blocked for security."
    return True, ""


def validate_document_content_security(content: str) -> Tuple[bool, str]:
    """Scan retrieved document content for embedded indirect prompt injection attacks."""
    for pattern in PROMPT_INJECTION_PATTERNS:
        if pattern.search(content):
            logger.warning("indirect_prompt_injection_detected_in_document")
            return False, "Retrieved document contains suspected prompt injection payload."
    return True, ""


def sanitize_output_security(answer_text: str) -> str:
    """Sanitize output text to ensure no sensitive API keys are leaked."""
    sanitized = answer_text
    for pattern in SENSITIVE_KEY_PATTERNS:
        sanitized = pattern.sub("[REDACTED_API_KEY]", sanitized)
    return sanitized

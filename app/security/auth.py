"""
Authentication and Rate Limiting for ARI API Endpoints.
"""
import time
from collections import defaultdict
from typing import Dict, List, Optional
from fastapi import Header, HTTPException, Request

from app.config.settings import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Simple in-memory sliding window rate limiter
# Key: client_ip or api_key, Value: list of timestamps
_request_history: Dict[str, List[float]] = defaultdict(list)
RATE_LIMIT_WINDOW_SECONDS = 60.0
RATE_LIMIT_MAX_REQUESTS = 60


async def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None),
) -> Optional[str]:
    """
    Verify API key against settings.api_key or API_KEY env variable if set.
    If no API key is configured in settings, allows unauthenticated requests in dev mode.
    """
    configured_key = getattr(settings, "api_key", None)
    if not configured_key:
        return None  # Auth disabled / open mode

    provided_key = x_api_key
    if not provided_key and authorization:
        if authorization.startswith("Bearer "):
            provided_key = authorization.split(" ", 1)[1]
        else:
            provided_key = authorization

    if not provided_key or provided_key != configured_key:
        logger.warning("unauthorized_api_access_attempt", extra={"provided": bool(provided_key)})
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return provided_key


async def check_rate_limit(request: Request) -> None:
    """
    Enforce in-memory sliding window rate limit per client IP.
    """
    client_ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    
    # Prune timestamps older than window
    timestamps = [t for t in _request_history[client_ip] if now - t < RATE_LIMIT_WINDOW_SECONDS]
    
    if len(timestamps) >= RATE_LIMIT_MAX_REQUESTS:
        logger.warning("rate_limit_exceeded", extra={"client_ip": client_ip, "count": len(timestamps)})
        raise HTTPException(
            status_code=429,
            detail="Too Many Requests. Rate limit exceeded. Please wait before sending more requests.",
        )
    
    timestamps.append(now)
    _request_history[client_ip] = timestamps

"""
Observability middleware for FastAPI.
Generates unique X-Run-ID request identifiers, tracks response latencies, and logs structured request summaries.
"""
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import current_run_id, get_logger

logger = get_logger(__name__)


class ObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        run_id = str(uuid.uuid4())
        current_run_id.set(run_id)

        start = time.monotonic()

        response = await call_next(request)

        latency_ms = round((time.monotonic() - start) * 1000, 1)

        response.headers["X-Run-ID"] = run_id
        response.headers["X-Response-Time-Ms"] = str(latency_ms)

        logger.info(
            "http_request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": latency_ms,
                "run_id": run_id,
            },
        )

        return response

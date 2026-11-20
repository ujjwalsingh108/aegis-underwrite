"""
HTTP hardening for the gateway: request IDs, security headers, access logs, rate limiting.
Pure ASGI/Starlette, no extra dependencies.
"""

import re
import threading
import time
import uuid
from collections import deque

from fastapi import Depends, HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from src.api.auth import AuthenticatedUser, get_current_user
from src.common.config import settings
from src.common.logger import get_logger, request_id_ctx

logger = get_logger("http")

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._\-]{1,64}$")

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    # Underwriting data is sensitive financial PII: never cache it.
    "Cache-Control": "no-store",
}


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assigns/propagates X-Request-ID, sets security headers and writes an access log line."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        incoming = request.headers.get("X-Request-ID", "")
        request_id = incoming if _REQUEST_ID_RE.match(incoming) else uuid.uuid4().hex
        token = request_id_ctx.set(request_id)
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
        finally:
            logger.info(
                "request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": status_code,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                },
            )
            request_id_ctx.reset(token)
        response.headers["X-Request-ID"] = request_id
        for header, value in SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        return response


class SlidingWindowRateLimiter:
    """In-process sliding-window limiter keyed by caller identity.

    Limits are per container. Modal can run several gateway containers, so the effective global
    limit is ``limit x containers``; use a shared store (e.g. Redis/Modal Dict) for a hard cap.
    """

    def __init__(self, limit: int, window_seconds: float = 60.0):
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> float | None:
        """Records a hit; returns seconds to wait if the limit is exceeded, else None."""
        now = time.monotonic()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return self.window - (now - hits[0])
            hits.append(now)
            return None

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


underwriting_limiter = SlidingWindowRateLimiter(settings.rate_limit_per_minute)


async def enforce_rate_limit(
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> None:
    retry_after = underwriting_limiter.check(current_user.user_id)
    if retry_after is not None:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded for underwriting requests.",
            headers={"Retry-After": str(max(1, int(retry_after)))},
        )

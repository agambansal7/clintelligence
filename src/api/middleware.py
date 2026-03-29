"""
TrialIntel API Middleware

Security, logging, and rate limiting middleware for the API.
"""

import hashlib
import logging
import time
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Callable, Dict, Optional, Tuple

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging all API requests and responses.

    Logs:
    - Request method, path, and query params
    - Request ID for tracing
    - Response status and timing
    - Client IP (with privacy considerations)
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate unique request ID
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id

        # Get client IP (handle proxies)
        client_ip = request.client.host if request.client else "unknown"
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()

        # Hash IP for privacy in logs
        ip_hash = hashlib.sha256(client_ip.encode()).hexdigest()[:8]

        # Log request
        start_time = time.time()
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} "
            f"client={ip_hash} query={dict(request.query_params)}"
        )

        # Process request
        try:
            response = await call_next(request)
            duration_ms = (time.time() - start_time) * 1000

            # Log response
            logger.info(
                f"[{request_id}] {request.method} {request.url.path} "
                f"status={response.status_code} duration={duration_ms:.2f}ms"
            )

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            return response

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(
                f"[{request_id}] {request.method} {request.url.path} "
                f"error={str(e)} duration={duration_ms:.2f}ms"
            )
            raise


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Token bucket rate limiting middleware.

    Limits requests per client IP within a sliding time window.
    """

    def __init__(
        self,
        app,
        requests_per_window: int = 100,
        window_seconds: int = 60,
        exclude_paths: Optional[list] = None,
    ):
        super().__init__(app)
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.exclude_paths = exclude_paths or ["/api/v1/health", "/docs", "/openapi.json"]

        # Store: {client_ip: [(timestamp, count), ...]}
        self._requests: Dict[str, list] = defaultdict(list)
        self._cleanup_interval = 60  # Cleanup old entries every 60 seconds
        self._last_cleanup = time.time()

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        client_ip = request.client.host if request.client else "unknown"
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        return client_ip

    def _cleanup_old_entries(self):
        """Remove expired rate limit entries."""
        current_time = time.time()
        if current_time - self._last_cleanup < self._cleanup_interval:
            return

        cutoff = current_time - self.window_seconds
        for ip in list(self._requests.keys()):
            self._requests[ip] = [
                (ts, count) for ts, count in self._requests[ip]
                if ts > cutoff
            ]
            if not self._requests[ip]:
                del self._requests[ip]

        self._last_cleanup = current_time

    def _check_rate_limit(self, client_ip: str) -> Tuple[bool, int, int]:
        """
        Check if client is within rate limits.

        Returns:
            Tuple of (is_allowed, remaining_requests, reset_time_seconds)
        """
        current_time = time.time()
        cutoff = current_time - self.window_seconds

        # Get recent requests for this IP
        recent_requests = [
            (ts, count) for ts, count in self._requests[client_ip]
            if ts > cutoff
        ]
        self._requests[client_ip] = recent_requests

        # Count total requests in window
        total_requests = sum(count for _, count in recent_requests)

        if total_requests >= self.requests_per_window:
            # Calculate reset time
            oldest_request = min(ts for ts, _ in recent_requests) if recent_requests else current_time
            reset_time = int(oldest_request + self.window_seconds - current_time)
            return False, 0, max(reset_time, 1)

        # Add current request
        self._requests[client_ip].append((current_time, 1))
        remaining = self.requests_per_window - total_requests - 1

        return True, remaining, self.window_seconds

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # Periodic cleanup
        self._cleanup_old_entries()

        # Check rate limit
        client_ip = self._get_client_ip(request)
        is_allowed, remaining, reset_time = self._check_rate_limit(client_ip)

        if not is_allowed:
            logger.warning(f"Rate limit exceeded for client: {hashlib.sha256(client_ip.encode()).hexdigest()[:8]}")
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded",
                    "retry_after_seconds": reset_time,
                },
                headers={
                    "Retry-After": str(reset_time),
                    "X-RateLimit-Limit": str(self.requests_per_window),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_time),
                },
            )

        # Process request and add rate limit headers
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_window)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_time)

        return response


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """
    API Key authentication middleware.

    Validates API keys passed in the X-API-Key header.
    """

    def __init__(
        self,
        app,
        api_keys: list,
        header_name: str = "X-API-Key",
        exclude_paths: Optional[list] = None,
        enabled: bool = True,
    ):
        super().__init__(app)
        self.api_keys = set(api_keys)
        self.header_name = header_name
        self.exclude_paths = exclude_paths or [
            "/",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/v1/health",
        ]
        self.enabled = enabled

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip if auth is disabled
        if not self.enabled:
            return await call_next(request)

        # Skip for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # Check API key
        api_key = request.headers.get(self.header_name)

        if not api_key:
            logger.warning(f"Missing API key for {request.url.path}")
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing API key. Include X-API-Key header."},
            )

        if api_key not in self.api_keys:
            # Hash the key for logging (don't log actual keys)
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:8]
            logger.warning(f"Invalid API key (hash: {key_hash}) for {request.url.path}")
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid API key"},
            )

        # Store validated key in request state
        request.state.api_key = api_key
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers to all responses.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        return response


def setup_middleware(app, settings):
    """
    Configure all middleware for the FastAPI application.

    Args:
        app: FastAPI application instance
        settings: Application settings
    """
    from fastapi.middleware.cors import CORSMiddleware

    # Order matters! Add in reverse order of execution

    # 1. Security headers (runs last, wraps response)
    app.add_middleware(SecurityHeadersMiddleware)

    # 2. Rate limiting
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_window=settings.api.rate_limit_requests,
        window_seconds=settings.api.rate_limit_window_seconds,
    )

    # 3. API Key authentication
    app.add_middleware(
        APIKeyAuthMiddleware,
        api_keys=settings.api.api_keys,
        header_name=settings.api.api_key_header,
        enabled=settings.api.require_api_key,
    )

    # 4. Request logging (runs first, logs everything)
    app.add_middleware(RequestLoggingMiddleware)

    # 5. CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=settings.api.cors_allow_credentials,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    logger.info(
        f"Middleware configured: "
        f"CORS origins={settings.api.cors_origins}, "
        f"rate_limit={settings.api.rate_limit_requests}/{settings.api.rate_limit_window_seconds}s, "
        f"api_key_auth={'enabled' if settings.api.require_api_key else 'disabled'}"
    )

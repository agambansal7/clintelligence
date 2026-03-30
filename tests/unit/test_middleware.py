"""
Tests for API middleware components.
"""

import pytest
import time
from unittest.mock import MagicMock, patch

from starlette.testclient import TestClient
from fastapi import FastAPI


class TestRateLimitMiddleware:
    """Tests for rate limiting middleware."""

    def test_allows_requests_under_limit(self):
        """Should allow requests under the rate limit."""
        from src.api.middleware import RateLimitMiddleware

        app = FastAPI()
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_window=10,
            window_seconds=60
        )

        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)

        # Should allow multiple requests
        for _ in range(5):
            response = client.get("/test")
            assert response.status_code == 200

    def test_blocks_requests_over_limit(self):
        """Should block requests over the rate limit."""
        from src.api.middleware import RateLimitMiddleware

        app = FastAPI()
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_window=3,
            window_seconds=60
        )

        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)

        # First 3 requests should succeed
        for _ in range(3):
            response = client.get("/test")
            assert response.status_code == 200

        # 4th request should be rate limited
        response = client.get("/test")
        assert response.status_code == 429
        assert "Rate limit exceeded" in response.json()["detail"]

    def test_includes_rate_limit_headers(self):
        """Should include rate limit headers in response."""
        from src.api.middleware import RateLimitMiddleware

        app = FastAPI()
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_window=10,
            window_seconds=60
        )

        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/test")

        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers

    def test_excludes_health_endpoint(self):
        """Should not rate limit health check endpoint."""
        from src.api.middleware import RateLimitMiddleware

        app = FastAPI()
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_window=1,
            window_seconds=60,
            exclude_paths=["/api/v1/health"]
        )

        @app.get("/api/v1/health")
        def health():
            return {"status": "healthy"}

        client = TestClient(app)

        # Multiple health requests should all succeed
        for _ in range(10):
            response = client.get("/api/v1/health")
            assert response.status_code == 200


class TestAPIKeyAuthMiddleware:
    """Tests for API key authentication middleware."""

    def test_allows_valid_api_key(self):
        """Should allow requests with valid API key."""
        from src.api.middleware import APIKeyAuthMiddleware

        app = FastAPI()
        app.add_middleware(
            APIKeyAuthMiddleware,
            api_keys=["valid-key-123"],
            enabled=True
        )

        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/test", headers={"X-API-Key": "valid-key-123"})
        assert response.status_code == 200

    def test_rejects_missing_api_key(self):
        """Should reject requests without API key."""
        from src.api.middleware import APIKeyAuthMiddleware

        app = FastAPI()
        app.add_middleware(
            APIKeyAuthMiddleware,
            api_keys=["valid-key-123"],
            enabled=True
        )

        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/test")
        assert response.status_code == 401
        assert "Missing API key" in response.json()["detail"]

    def test_rejects_invalid_api_key(self):
        """Should reject requests with invalid API key."""
        from src.api.middleware import APIKeyAuthMiddleware

        app = FastAPI()
        app.add_middleware(
            APIKeyAuthMiddleware,
            api_keys=["valid-key-123"],
            enabled=True
        )

        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/test", headers={"X-API-Key": "invalid-key"})
        assert response.status_code == 401
        assert "Invalid API key" in response.json()["detail"]

    def test_bypasses_when_disabled(self):
        """Should bypass auth when disabled."""
        from src.api.middleware import APIKeyAuthMiddleware

        app = FastAPI()
        app.add_middleware(
            APIKeyAuthMiddleware,
            api_keys=["valid-key-123"],
            enabled=False
        )

        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/test")  # No API key
        assert response.status_code == 200

    def test_excludes_docs_endpoint(self):
        """Should not require API key for docs."""
        from src.api.middleware import APIKeyAuthMiddleware

        app = FastAPI()
        app.add_middleware(
            APIKeyAuthMiddleware,
            api_keys=["valid-key-123"],
            enabled=True,
            exclude_paths=["/", "/docs", "/openapi.json"]
        )

        @app.get("/")
        def root():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200


class TestRequestLoggingMiddleware:
    """Tests for request logging middleware."""

    def test_adds_request_id_header(self):
        """Should add X-Request-ID header to response."""
        from src.api.middleware import RequestLoggingMiddleware

        app = FastAPI()
        app.add_middleware(RequestLoggingMiddleware)

        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/test")

        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) == 8


class TestSecurityHeadersMiddleware:
    """Tests for security headers middleware."""

    def test_adds_security_headers(self):
        """Should add security headers to response."""
        from src.api.middleware import SecurityHeadersMiddleware

        app = FastAPI()
        app.add_middleware(SecurityHeadersMiddleware)

        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/test")

        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert response.headers.get("X-Frame-Options") == "DENY"
        assert response.headers.get("X-XSS-Protection") == "1; mode=block"
        assert "Strict-Transport-Security" in response.headers

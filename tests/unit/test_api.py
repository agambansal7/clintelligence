"""Unit tests for API endpoints.

Note: These are basic endpoint tests that verify the API responds correctly.
Full integration tests should be run with a populated database.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client for API."""
    from src.api.main import app
    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_root_endpoint(self, client):
        """Test root endpoint returns valid response."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        # Root should return some info about the service
        assert "service" in data or "status" in data or "version" in data


class TestStatsEndpoints:
    """Tests for stats endpoints."""

    def test_get_stats_returns_response(self, client):
        """Test stats endpoint returns a response."""
        response = client.get("/api/v1/stats")
        assert response.status_code == 200
        # May have total_trials or a message about empty DB
        assert isinstance(response.json(), dict)


class TestProtocolEndpoints:
    """Tests for protocol scoring endpoints."""

    def test_score_protocol_validation(self, client):
        """Test protocol scoring with invalid data returns 422."""
        # Missing required fields
        response = client.post(
            "/api/v1/protocol/risk-score",
            json={"condition": "diabetes"},  # Missing other required fields
        )
        # Should return 422 for validation error
        assert response.status_code == 422


class TestAIAssistantEndpoints:
    """Tests for AI assistant endpoints."""

    def test_ai_assistant_without_api_key(self, client, monkeypatch):
        """Test AI assistant returns 503 without API key."""
        # Ensure no API key is set
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        response = client.post(
            "/api/v1/assistant/query",
            json={"query": "What are typical diabetes trial endpoints?"},
        )
        # Should return 503 if API key not set (or anthropic not installed)
        assert response.status_code == 503

    def test_ai_assistant_request_validation(self, client):
        """Test AI assistant validates request body."""
        # Missing required query field
        response = client.post(
            "/api/v1/assistant/query",
            json={},
        )
        # Should return 422 for validation error
        assert response.status_code == 422

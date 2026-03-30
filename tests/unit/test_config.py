"""
Tests for configuration management.
"""

import os
import pytest
from unittest.mock import patch


class TestDatabaseConfig:
    """Tests for DatabaseConfig."""

    def test_default_sqlite_url(self):
        """Should use default SQLite URL."""
        from src.config import DatabaseConfig

        with patch.dict(os.environ, {}, clear=True):
            config = DatabaseConfig()
            assert "sqlite:///" in config.url

    def test_custom_database_url(self):
        """Should use DATABASE_URL from environment."""
        from src.config import DatabaseConfig

        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://user:pass@localhost/db"}):
            config = DatabaseConfig()
            assert config.url == "postgresql://user:pass@localhost/db"

    def test_pool_settings(self):
        """Should read pool settings from environment."""
        from src.config import DatabaseConfig

        with patch.dict(os.environ, {"DB_POOL_SIZE": "10", "DB_MAX_OVERFLOW": "20"}):
            config = DatabaseConfig()
            assert config.pool_size == 10
            assert config.max_overflow == 20


class TestAPIConfig:
    """Tests for APIConfig."""

    def test_default_host_and_port(self):
        """Should use default host and port."""
        from src.config import APIConfig

        with patch.dict(os.environ, {}, clear=True):
            config = APIConfig()
            assert config.host == "0.0.0.0"
            assert config.port == 8000

    def test_custom_host_and_port(self):
        """Should use custom host and port from environment."""
        from src.config import APIConfig

        with patch.dict(os.environ, {"API_HOST": "127.0.0.1", "API_PORT": "9000"}):
            config = APIConfig()
            assert config.host == "127.0.0.1"
            assert config.port == 9000

    def test_cors_origins_parsing(self):
        """Should parse CORS origins from comma-separated string."""
        from src.config import APIConfig

        with patch.dict(os.environ, {"CORS_ORIGINS": "http://localhost:3000,http://example.com"}):
            config = APIConfig()
            assert "http://localhost:3000" in config.cors_origins
            assert "http://example.com" in config.cors_origins

    def test_rate_limiting_settings(self):
        """Should read rate limiting settings."""
        from src.config import APIConfig

        with patch.dict(os.environ, {"RATE_LIMIT_REQUESTS": "50", "RATE_LIMIT_WINDOW": "120"}):
            config = APIConfig()
            assert config.rate_limit_requests == 50
            assert config.rate_limit_window_seconds == 120

    def test_api_key_settings(self):
        """Should read API key settings."""
        from src.config import APIConfig

        with patch.dict(os.environ, {"REQUIRE_API_KEY": "true", "API_KEYS": "key1,key2,key3"}):
            config = APIConfig()
            assert config.require_api_key is True
            assert len(config.api_keys) == 3
            assert "key1" in config.api_keys


class TestAIConfig:
    """Tests for AIConfig."""

    def test_anthropic_api_key(self):
        """Should read Anthropic API key."""
        from src.config import AIConfig

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key-123"}):
            config = AIConfig()
            assert config.anthropic_api_key == "test-key-123"

    def test_default_model(self):
        """Should use default model."""
        from src.config import AIConfig

        with patch.dict(os.environ, {}, clear=True):
            config = AIConfig()
            assert "claude" in config.model.lower()

    def test_custom_model(self):
        """Should allow custom model."""
        from src.config import AIConfig

        with patch.dict(os.environ, {"CLAUDE_MODEL": "claude-3-opus-20240229"}):
            config = AIConfig()
            assert config.model == "claude-3-opus-20240229"


class TestCacheConfig:
    """Tests for CacheConfig."""

    def test_default_enabled(self):
        """Should be enabled by default."""
        from src.config import CacheConfig

        with patch.dict(os.environ, {}, clear=True):
            config = CacheConfig()
            assert config.enabled is True

    def test_disable_cache(self):
        """Should allow disabling cache."""
        from src.config import CacheConfig

        with patch.dict(os.environ, {"CACHE_ENABLED": "false"}):
            config = CacheConfig()
            assert config.enabled is False

    def test_ttl_setting(self):
        """Should read TTL setting."""
        from src.config import CacheConfig

        with patch.dict(os.environ, {"CACHE_TTL": "600"}):
            config = CacheConfig()
            assert config.ttl_seconds == 600

    def test_redis_url(self):
        """Should read Redis URL."""
        from src.config import CacheConfig

        with patch.dict(os.environ, {"REDIS_URL": "redis://localhost:6379/1"}):
            config = CacheConfig()
            assert config.redis_url == "redis://localhost:6379/1"


class TestSettings:
    """Tests for main Settings class."""

    def test_environment_detection(self):
        """Should detect environment."""
        from src.config import Settings

        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            settings = Settings()
            assert settings.is_production
            assert not settings.is_development

        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            settings = Settings()
            assert settings.is_development
            assert not settings.is_production

        with patch.dict(os.environ, {"ENVIRONMENT": "testing"}):
            settings = Settings()
            assert settings.is_testing

    def test_nested_configs(self):
        """Should create nested config objects."""
        from src.config import Settings

        settings = Settings()
        assert settings.database is not None
        assert settings.api is not None
        assert settings.ai is not None
        assert settings.cache is not None
        assert settings.logging is not None


class TestGetSettings:
    """Tests for get_settings function."""

    def test_returns_settings_instance(self):
        """Should return Settings instance."""
        from src.config import get_settings

        # Clear cache first
        get_settings.cache_clear()

        settings = get_settings()
        assert settings is not None
        assert hasattr(settings, 'database')
        assert hasattr(settings, 'api')

    def test_caches_result(self):
        """Should cache settings instance."""
        from src.config import get_settings

        get_settings.cache_clear()

        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

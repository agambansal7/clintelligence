"""
TrialIntel Configuration Management

Centralized configuration with environment variable support and validation.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from functools import lru_cache


@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str = field(default_factory=lambda: os.getenv(
        "DATABASE_URL",
        f"sqlite:///{os.getenv('TRIALINTEL_DATA_DIR', './data')}/trials.db"
    ))
    pool_size: int = field(default_factory=lambda: int(os.getenv("DB_POOL_SIZE", "5")))
    max_overflow: int = field(default_factory=lambda: int(os.getenv("DB_MAX_OVERFLOW", "10")))
    echo: bool = field(default_factory=lambda: os.getenv("DB_ECHO", "false").lower() == "true")


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    debug: bool = field(default_factory=lambda: os.getenv("API_DEBUG", "false").lower() == "true")
    reload: bool = field(default_factory=lambda: os.getenv("API_RELOAD", "false").lower() == "true")

    # CORS settings
    cors_origins: List[str] = field(default_factory=lambda: [
        origin.strip() for origin in
        os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8501").split(",")
        if origin.strip()
    ])
    cors_allow_credentials: bool = field(
        default_factory=lambda: os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true"
    )

    # Rate limiting
    rate_limit_requests: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_REQUESTS", "100")))
    rate_limit_window_seconds: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_WINDOW", "60")))

    # API Authentication
    api_key_header: str = "X-API-Key"
    require_api_key: bool = field(
        default_factory=lambda: os.getenv("REQUIRE_API_KEY", "false").lower() == "true"
    )
    api_keys: List[str] = field(default_factory=lambda: [
        key.strip() for key in
        os.getenv("API_KEYS", "").split(",")
        if key.strip()
    ])


@dataclass
class AIConfig:
    """AI/LLM configuration."""
    anthropic_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )
    model: str = field(default_factory=lambda: os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514"))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("CLAUDE_MAX_TOKENS", "1024")))


@dataclass
class CacheConfig:
    """Cache configuration."""
    enabled: bool = field(default_factory=lambda: os.getenv("CACHE_ENABLED", "true").lower() == "true")
    ttl_seconds: int = field(default_factory=lambda: int(os.getenv("CACHE_TTL", "300")))
    redis_url: Optional[str] = field(default_factory=lambda: os.getenv("REDIS_URL"))


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    format: str = field(default_factory=lambda: os.getenv(
        "LOG_FORMAT",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    json_format: bool = field(default_factory=lambda: os.getenv("LOG_JSON", "false").lower() == "true")
    file_path: Optional[str] = field(default_factory=lambda: os.getenv("LOG_FILE"))


@dataclass
class IngestionConfig:
    """Data ingestion configuration."""
    ctgov_base_url: str = "https://clinicaltrials.gov/api/v2"
    rate_limit_delay: float = field(default_factory=lambda: float(os.getenv("CTGOV_RATE_LIMIT", "0.1")))
    batch_size: int = field(default_factory=lambda: int(os.getenv("INGESTION_BATCH_SIZE", "100")))


@dataclass
class Settings:
    """Main settings container."""
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))

    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    api: APIConfig = field(default_factory=APIConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"

    @property
    def is_development(self) -> bool:
        return self.environment.lower() == "development"

    @property
    def is_testing(self) -> bool:
        return self.environment.lower() == "testing"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    # Load .env file if it exists
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # python-dotenv not installed, use environment variables directly

    return Settings()


# Convenience function for direct access
settings = get_settings()

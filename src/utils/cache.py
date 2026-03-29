"""
TrialIntel Caching Layer

Provides caching for API responses and analysis results with support for:
- In-memory caching (default)
- Redis caching (optional, for distributed deployments)
"""

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class CacheEntry:
    """Represents a cached value with metadata."""
    value: Any
    created_at: float
    ttl: int  # seconds
    hits: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return time.time() > self.created_at + self.ttl

    @property
    def age_seconds(self) -> float:
        """Get the age of the cache entry in seconds."""
        return time.time() - self.created_at


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """Set a value in cache with TTL."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a value from cache."""
        pass

    @abstractmethod
    def clear(self) -> bool:
        """Clear all cached values."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class InMemoryCache(CacheBackend):
    """
    Thread-safe in-memory cache implementation.

    Suitable for single-instance deployments.
    """

    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, CacheEntry] = {}
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
        self._lock = None  # Use threading.Lock() for thread safety

    def _ensure_lock(self):
        """Lazily create lock to avoid import issues."""
        if self._lock is None:
            import threading
            self._lock = threading.Lock()
        return self._lock

    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        lock = self._ensure_lock()
        with lock:
            entry = self._cache.get(key)

            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired:
                del self._cache[key]
                self._misses += 1
                return None

            entry.hits += 1
            self._hits += 1
            return entry.value

    def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """Set a value in cache with TTL."""
        lock = self._ensure_lock()
        with lock:
            # Evict expired entries if at capacity
            if len(self._cache) >= self._max_size:
                self._evict_expired()

            # If still at capacity, evict least recently accessed
            if len(self._cache) >= self._max_size:
                self._evict_lru()

            self._cache[key] = CacheEntry(
                value=value,
                created_at=time.time(),
                ttl=ttl
            )
            return True

    def delete(self, key: str) -> bool:
        """Delete a value from cache."""
        lock = self._ensure_lock()
        with lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> bool:
        """Clear all cached values."""
        lock = self._ensure_lock()
        with lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            return True

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        lock = self._ensure_lock()
        with lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0

            return {
                "backend": "memory",
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 3),
            }

    def _evict_expired(self):
        """Remove all expired entries."""
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired
        ]
        for key in expired_keys:
            del self._cache[key]

    def _evict_lru(self):
        """Remove least recently used entry."""
        if not self._cache:
            return

        # Find oldest entry (by creation time + hits as proxy for recency)
        oldest_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].created_at + self._cache[k].hits
        )
        del self._cache[oldest_key]


class RedisCache(CacheBackend):
    """
    Redis-backed cache implementation.

    Suitable for distributed deployments.
    """

    def __init__(self, url: str = "redis://localhost:6379/0", prefix: str = "trialintel:"):
        self._url = url
        self._prefix = prefix
        self._client = None
        self._connected = False

    def _get_client(self):
        """Get or create Redis client."""
        if self._client is None:
            try:
                import redis
                self._client = redis.from_url(self._url, decode_responses=True)
                # Test connection
                self._client.ping()
                self._connected = True
                logger.info(f"Connected to Redis at {self._url}")
            except ImportError:
                logger.error("Redis package not installed. Install with: pip install redis")
                raise
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self._connected = False
                raise
        return self._client

    def _make_key(self, key: str) -> str:
        """Create prefixed cache key."""
        return f"{self._prefix}{key}"

    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        try:
            client = self._get_client()
            full_key = self._make_key(key)
            data = client.get(full_key)

            if data is None:
                return None

            return json.loads(data)
        except Exception as e:
            logger.warning(f"Redis get error: {e}")
            return None

    def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """Set a value in cache with TTL."""
        try:
            client = self._get_client()
            full_key = self._make_key(key)
            data = json.dumps(value, default=str)
            client.setex(full_key, ttl, data)
            return True
        except Exception as e:
            logger.warning(f"Redis set error: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete a value from cache."""
        try:
            client = self._get_client()
            full_key = self._make_key(key)
            return bool(client.delete(full_key))
        except Exception as e:
            logger.warning(f"Redis delete error: {e}")
            return False

    def clear(self) -> bool:
        """Clear all cached values with our prefix."""
        try:
            client = self._get_client()
            pattern = f"{self._prefix}*"
            keys = client.keys(pattern)
            if keys:
                client.delete(*keys)
            return True
        except Exception as e:
            logger.warning(f"Redis clear error: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            client = self._get_client()
            info = client.info("stats")
            pattern = f"{self._prefix}*"
            key_count = len(client.keys(pattern))

            return {
                "backend": "redis",
                "connected": self._connected,
                "keys": key_count,
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "memory_used": info.get("used_memory_human", "unknown"),
            }
        except Exception as e:
            return {
                "backend": "redis",
                "connected": False,
                "error": str(e),
            }


class CacheManager:
    """
    Main cache manager that provides a unified interface.

    Automatically selects backend based on configuration.
    """

    _instance: Optional['CacheManager'] = None

    def __init__(
        self,
        backend: Optional[CacheBackend] = None,
        redis_url: Optional[str] = None,
        enabled: bool = True,
        default_ttl: int = 300,
    ):
        self._enabled = enabled
        self._default_ttl = default_ttl

        if backend:
            self._backend = backend
        elif redis_url:
            try:
                self._backend = RedisCache(url=redis_url)
            except Exception:
                logger.warning("Redis unavailable, falling back to in-memory cache")
                self._backend = InMemoryCache()
        else:
            self._backend = InMemoryCache()

    @classmethod
    def get_instance(cls) -> 'CacheManager':
        """Get or create singleton instance."""
        if cls._instance is None:
            import os
            redis_url = os.getenv("REDIS_URL")
            enabled = os.getenv("CACHE_ENABLED", "true").lower() == "true"
            default_ttl = int(os.getenv("CACHE_TTL", "300"))

            cls._instance = cls(
                redis_url=redis_url,
                enabled=enabled,
                default_ttl=default_ttl,
            )
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance."""
        if cls._instance:
            cls._instance.clear()
        cls._instance = None

    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        if not self._enabled:
            return None
        return self._backend.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in cache."""
        if not self._enabled:
            return False
        return self._backend.set(key, value, ttl or self._default_ttl)

    def delete(self, key: str) -> bool:
        """Delete a value from cache."""
        return self._backend.delete(key)

    def clear(self) -> bool:
        """Clear all cached values."""
        return self._backend.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self._backend.get_stats()
        stats["enabled"] = self._enabled
        stats["default_ttl"] = self._default_ttl
        return stats

    def cached(
        self,
        ttl: Optional[int] = None,
        key_prefix: str = "",
    ) -> Callable:
        """
        Decorator to cache function results.

        Usage:
            @cache_manager.cached(ttl=300)
            def expensive_function(arg1, arg2):
                ...
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            def wrapper(*args, **kwargs) -> T:
                if not self._enabled:
                    return func(*args, **kwargs)

                # Generate cache key
                cache_key = self._make_cache_key(func, args, kwargs, key_prefix)

                # Try to get from cache
                cached_value = self.get(cache_key)
                if cached_value is not None:
                    logger.debug(f"Cache hit: {cache_key}")
                    return cached_value

                # Execute function and cache result
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl)
                logger.debug(f"Cache miss, stored: {cache_key}")

                return result

            return wrapper
        return decorator

    def _make_cache_key(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        prefix: str = "",
    ) -> str:
        """Generate a unique cache key for function call."""
        # Serialize arguments
        key_parts = [
            prefix or func.__module__,
            func.__name__,
            str(args),
            str(sorted(kwargs.items())),
        ]
        key_string = ":".join(key_parts)

        # Hash for consistent key length
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        return f"{prefix or func.__name__}:{key_hash}"


# Convenience function for getting the global cache manager
def get_cache() -> CacheManager:
    """Get the global cache manager instance."""
    return CacheManager.get_instance()


# Decorator for caching function results
def cached(ttl: int = 300, key_prefix: str = "") -> Callable:
    """
    Decorator to cache function results.

    Usage:
        @cached(ttl=300)
        def expensive_function(arg1, arg2):
            ...
    """
    return get_cache().cached(ttl=ttl, key_prefix=key_prefix)

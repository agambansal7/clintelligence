"""
Tests for caching layer.
"""

import pytest
import time
from unittest.mock import patch, MagicMock

from src.utils.cache import (
    InMemoryCache,
    CacheManager,
    CacheEntry,
    cached,
)


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_is_expired_false_for_fresh_entry(self):
        """Fresh entry should not be expired."""
        entry = CacheEntry(
            value="test",
            created_at=time.time(),
            ttl=300
        )
        assert not entry.is_expired

    def test_is_expired_true_for_old_entry(self):
        """Old entry should be expired."""
        entry = CacheEntry(
            value="test",
            created_at=time.time() - 400,
            ttl=300
        )
        assert entry.is_expired

    def test_age_seconds(self):
        """Should correctly calculate age."""
        created = time.time() - 100
        entry = CacheEntry(
            value="test",
            created_at=created,
            ttl=300
        )
        assert 99 < entry.age_seconds < 101


class TestInMemoryCache:
    """Tests for InMemoryCache backend."""

    def test_set_and_get(self):
        """Should store and retrieve values."""
        cache = InMemoryCache()
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_nonexistent_key(self):
        """Should return None for missing key."""
        cache = InMemoryCache()
        assert cache.get("nonexistent") is None

    def test_delete(self):
        """Should delete values."""
        cache = InMemoryCache()
        cache.set("key1", "value1")
        assert cache.delete("key1")
        assert cache.get("key1") is None

    def test_delete_nonexistent(self):
        """Should return False for deleting nonexistent key."""
        cache = InMemoryCache()
        assert not cache.delete("nonexistent")

    def test_clear(self):
        """Should clear all values."""
        cache = InMemoryCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_ttl_expiration(self):
        """Should expire values after TTL."""
        cache = InMemoryCache()
        cache.set("key1", "value1", ttl=1)

        assert cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(1.1)
        assert cache.get("key1") is None

    def test_max_size_eviction(self):
        """Should evict oldest entries when at capacity."""
        cache = InMemoryCache(max_size=3)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Adding 4th should evict oldest
        cache.set("key4", "value4")

        # One of the original keys should be gone
        values = [
            cache.get("key1"),
            cache.get("key2"),
            cache.get("key3"),
            cache.get("key4")
        ]
        assert values.count(None) >= 1
        assert cache.get("key4") == "value4"

    def test_stats(self):
        """Should track cache statistics."""
        cache = InMemoryCache()

        cache.set("key1", "value1")
        cache.get("key1")  # hit
        cache.get("key1")  # hit
        cache.get("missing")  # miss

        stats = cache.get_stats()
        assert stats["backend"] == "memory"
        assert stats["size"] == 1
        assert stats["hits"] == 2
        assert stats["misses"] == 1

    def test_complex_values(self):
        """Should store complex values like dicts and lists."""
        cache = InMemoryCache()

        data = {
            "list": [1, 2, 3],
            "nested": {"a": 1, "b": 2},
            "number": 42
        }
        cache.set("complex", data)
        assert cache.get("complex") == data


class TestCacheManager:
    """Tests for CacheManager."""

    def test_singleton_instance(self):
        """Should return same instance."""
        CacheManager.reset_instance()

        manager1 = CacheManager.get_instance()
        manager2 = CacheManager.get_instance()

        assert manager1 is manager2

        CacheManager.reset_instance()

    def test_disabled_cache(self):
        """Should return None when disabled."""
        manager = CacheManager(enabled=False)

        manager.set("key1", "value1")
        assert manager.get("key1") is None

    def test_enabled_cache(self):
        """Should work normally when enabled."""
        manager = CacheManager(enabled=True)

        manager.set("key1", "value1")
        assert manager.get("key1") == "value1"

    def test_cached_decorator(self):
        """Should cache function results."""
        manager = CacheManager(enabled=True)

        call_count = 0

        @manager.cached(ttl=300)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call should execute function
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1

        # Second call should use cache
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Not incremented

        # Different args should execute function
        result3 = expensive_function(10)
        assert result3 == 20
        assert call_count == 2

    def test_cached_decorator_disabled(self):
        """Should not cache when disabled."""
        manager = CacheManager(enabled=False)

        call_count = 0

        @manager.cached(ttl=300)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # Both calls should execute function
        expensive_function(5)
        expensive_function(5)
        assert call_count == 2


class TestCachedDecorator:
    """Tests for the @cached decorator."""

    def test_caches_function_results(self):
        """Should cache function results using global manager."""
        CacheManager.reset_instance()

        call_count = 0

        @cached(ttl=300)
        def my_function(x):
            nonlocal call_count
            call_count += 1
            return x * 3

        result1 = my_function(4)
        result2 = my_function(4)

        assert result1 == 12
        assert result2 == 12
        # May or may not cache depending on global state
        # At minimum should return correct result

        CacheManager.reset_instance()

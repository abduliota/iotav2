"""Prompt / response cache abstraction for the RAG pipeline.

This module provides a thin wrapper over an in-memory or Redis-backed cache
so that expensive end-to-end RAG+LLM calls can be reused when the effective
context has not changed.

The cache operates on *logical context* rather than raw prompt strings:
callers are expected to pass a small, stable dictionary that captures the
essentials of the request (query, intent, retrieved document IDs/pages,
memory IDs, routing info, etc.). This dictionary is then serialized and
hashed into a cache key.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from config import (
    CACHE_BACKEND,
    CACHE_ENABLED,
    CACHE_TTL_SECONDS,
    REDIS_URL,
)

log = logging.getLogger("prompt_cache")


def _stable_dumps(obj: Dict[str, Any]) -> str:
    """Stable JSON serialization for cache key material."""

    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _hash_key(payload: Dict[str, Any]) -> str:
    """Return a deterministic hash for the given payload."""

    raw = _stable_dumps(payload).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass
class CacheEntry:
    value: Dict[str, Any]
    expires_at: float


class BasePromptCache:
    """Abstract cache interface used by the RAG pipeline."""

    def __init__(self, ttl_seconds: int) -> None:
        self.ttl_seconds = ttl_seconds

    @property
    def enabled(self) -> bool:
        return self.ttl_seconds > 0

    def build_key(self, context_payload: Dict[str, Any]) -> str:
        """Build a namespaced cache key from a logical context payload."""

        return f"prompt:{_hash_key(context_payload)}"

    def get(self, key: str) -> Optional[Dict[str, Any]]:  # pragma: no cover - interface
        raise NotImplementedError

    def set(self, key: str, value: Dict[str, Any]) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class InMemoryPromptCache(BasePromptCache):
    """Simple process-local cache suitable for development and single-instance runs."""

    def __init__(self, ttl_seconds: int) -> None:
        super().__init__(ttl_seconds)
        self._store: Dict[str, CacheEntry] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        now = time.time()
        with self._lock:
            entry = self._store.get(key)
            if not entry:
                return None
            if entry.expires_at <= now:
                self._store.pop(key, None)
                return None
            return entry.value

    def set(self, key: str, value: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        expires_at = time.time() + self.ttl_seconds
        with self._lock:
            self._store[key] = CacheEntry(value=value, expires_at=expires_at)


class RedisPromptCache(BasePromptCache):
    """Redis-backed cache; requires the `redis` Python package and REDIS_URL."""

    def __init__(self, ttl_seconds: int, redis_url: str) -> None:
        super().__init__(ttl_seconds)
        self._redis = None
        self._redis_url = redis_url.strip()
        if not self._redis_url:
            log.warning("RedisPromptCache: REDIS_URL not configured; cache disabled.")
            return
        try:
            import redis  # type: ignore[import]

            self._redis = redis.Redis.from_url(self._redis_url)
        except Exception as exc:  # pragma: no cover - best-effort
            log.warning("RedisPromptCache: could not initialize Redis client: %s", exc)
            self._redis = None

    @property
    def enabled(self) -> bool:
        return super().enabled and self._redis is not None

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.enabled or self._redis is None:
            return None
        try:
            raw = self._redis.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("RedisPromptCache.get failed for key=%s: %s", key, exc)
            return None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        if not self.enabled or self._redis is None:
            return
        try:
            self._redis.setex(key, self.ttl_seconds, json.dumps(value))
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("RedisPromptCache.set failed for key=%s: %s", key, exc)


_CACHE_SINGLETON: Optional[BasePromptCache] = None
_CACHE_LOCK = threading.Lock()


def get_prompt_cache() -> Optional[BasePromptCache]:
    """Return a singleton prompt cache instance based on config.

    When CACHE_ENABLED is False, this returns None and callers should skip
    caching entirely.
    """

    global _CACHE_SINGLETON
    if not CACHE_ENABLED or CACHE_TTL_SECONDS <= 0:
        return None
    if _CACHE_SINGLETON is not None:
        return _CACHE_SINGLETON
    with _CACHE_LOCK:
        if _CACHE_SINGLETON is not None:
            return _CACHE_SINGLETON
        backend = (CACHE_BACKEND or "memory").strip().lower()
        if backend == "redis":
            _CACHE_SINGLETON = RedisPromptCache(CACHE_TTL_SECONDS, REDIS_URL)
        else:
            _CACHE_SINGLETON = InMemoryPromptCache(CACHE_TTL_SECONDS)
        return _CACHE_SINGLETON


def build_cache_payload(
    *,
    query: str,
    intent: str,
    in_scope: bool,
    route: str | None,
    doc_keys: list[tuple[str, int, int]],
    memory_keys: list[str],
    profile_fingerprint: str | None,
) -> Dict[str, Any]:
    """Build a small, stable payload representing the logical context of a query.

    This is intentionally conservative: it avoids including raw text, but
    captures enough structure to approximate when the effective prompt would
    be the same (or very similar) as a prior call.
    """

    return {
        "q": query.strip(),
        "intent": intent,
        "in_scope": bool(in_scope),
        "route": route or "",
        "docs": doc_keys,
        "mem": memory_keys,
        "profile": profile_fingerprint or "",
    }


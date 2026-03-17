"""
IOTAV3 prompt / response cache abstraction.

This is a slimmed‑down version of the legacy cache module, wired to the
IOTAV3 config. It can use an in‑memory dictionary or Redis as a backend.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from IOTAV3.backend import config_iotav3 as cfg

log = logging.getLogger("iotav3.prompt_cache")


def _stable_dumps(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _hash_key(payload: Dict[str, Any]) -> str:
    raw = _stable_dumps(payload).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass
class CacheEntry:
    value: Dict[str, Any]
    expires_at: float


class BasePromptCache:
    def __init__(self, ttl_seconds: int) -> None:
        self.ttl_seconds = ttl_seconds

    @property
    def enabled(self) -> bool:
        return self.ttl_seconds > 0

    def build_key(self, context_payload: Dict[str, Any]) -> str:
        return f"iotav3:prompt:{_hash_key(context_payload)}"

    def get(self, key: str) -> Optional[Dict[str, Any]]:  # pragma: no cover - interface
        raise NotImplementedError

    def set(self, key: str, value: Dict[str, Any]) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class InMemoryPromptCache(BasePromptCache):
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
    """
    Return a singleton prompt cache instance based on IOTAV3 config.
    """

    global _CACHE_SINGLETON
    if not cfg.CACHE_ENABLED or cfg.CACHE_TTL_SECONDS <= 0:
        return None
    if _CACHE_SINGLETON is not None:
        return _CACHE_SINGLETON
    with _CACHE_LOCK:
        if _CACHE_SINGLETON is not None:
            return _CACHE_SINGLETON
        backend = (cfg.CACHE_BACKEND or "memory").strip().lower()
        if backend == "redis":
            _CACHE_SINGLETON = RedisPromptCache(cfg.CACHE_TTL_SECONDS, cfg.REDIS_URL)
        else:
            _CACHE_SINGLETON = InMemoryPromptCache(cfg.CACHE_TTL_SECONDS)
        return _CACHE_SINGLETON


def build_cache_payload(
    *,
    query: str,
    intent: str,
    in_scope: bool,
    route: str | None,
    doc_keys: List[Tuple[str, int, int]],
    memory_keys: List[str],
    profile_fingerprint: str | None,
) -> Dict[str, Any]:
    """
    Build a small, stable payload representing the logical context of a query.
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



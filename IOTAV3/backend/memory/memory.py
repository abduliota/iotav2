"""
IOTAV3 memory wrapper.

This module exposes the same public functions as the main `backend/memory.py`
module, but lives under the IOTAV3 namespace so that the IOTAV3 backend can
import from a local path without modifying the original implementation.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Dict, Optional

from memory import (  # type: ignore
    ENABLE_EPISODIC_MEMORY_WRITES,
    ENABLE_MEMORY_SYSTEM,
    EMBEDDING_DIMENSION,
    MEMORY_ITEM_MIN_LENGTH,
    MEMORY_MAX_CHARS,
    MEMORY_TOP_K,
    SESSION_SUMMARY_UPDATE_EVERY_N_MESSAGES,
    embed_memory_item,
    get_session_summary,
    get_user_profile,
    insert_memory_item,
    maybe_update_session_summary,
    maybe_write_episodic_from_exchange,
    search_memory_items,
    upsert_session_summary,
    upsert_user_profile,
    update_profile_from_exchange,
)

__all__ = [
    "ENABLE_EPISODIC_MEMORY_WRITES",
    "ENABLE_MEMORY_SYSTEM",
    "EMBEDDING_DIMENSION",
    "MEMORY_ITEM_MIN_LENGTH",
    "MEMORY_MAX_CHARS",
    "MEMORY_TOP_K",
    "SESSION_SUMMARY_UPDATE_EVERY_N_MESSAGES",
    "get_user_profile",
    "upsert_user_profile",
    "get_session_summary",
    "upsert_session_summary",
    "insert_memory_item",
    "embed_memory_item",
    "search_memory_items",
    "update_profile_from_exchange",
    "maybe_write_episodic_from_exchange",
    "maybe_update_session_summary",
]


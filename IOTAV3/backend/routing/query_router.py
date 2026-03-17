"""
IOTAV3 query router wrapper.

Re-exports the simple heuristic router from the main backend.
"""

from __future__ import annotations

from typing import Dict

from routing.query_router import route as _route  # type: ignore

__all__ = ["route"]


def route(query: str, metadata: Dict[str, object] | None = None) -> str:
    return _route(query, metadata)


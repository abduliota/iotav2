"""Heuristic query router for the RAG pipeline.

The router classifies each incoming query into a coarse route label that can
be used to adjust retrieval depth, memory usage, or even bypass RAG entirely
for purely conversational cases.

Routes are intentionally simple for now:
- \"rag\": standard RAG + memory flow (default).
- \"cag_only\": use cached knowledge / system instructions only.
- \"hybrid\": RAG + memory with stronger emphasis on conversational history.
"""

from __future__ import annotations

from typing import Dict


def route(query: str, metadata: Dict[str, object] | None = None) -> str:
    """Return a simple route label for the given query.

    The current implementation is deliberately conservative: it primarily
    distinguishes pure social/greeting queries from regulatory ones, and
    otherwise returns \"rag\". This avoids surprising behavior while still
    giving us a hook for future, ML-based routing.
    """

    if not query or not query.strip():
        return "rag"

    q = query.strip().lower()

    # Very simple greetings / chit-chat detection
    greetings = {
        "hi",
        "hello",
        "hey",
        "hi!",
        "hello!",
        "hey!",
        "مرحبا",
    }
    if q.rstrip(".!?") in greetings:
        return "cag_only"

    # If the caller already computed intent, we can treat some intents as
    # more conversational and nudge toward a hybrid path.
    intent = ""
    if metadata and isinstance(metadata.get("intent"), str):
        intent = str(metadata.get("intent") or "")

    if intent in {"procedural", "synthesis"}:
        return "hybrid"

    return "rag"


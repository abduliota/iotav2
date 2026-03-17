"""
Standalone structured context builder for IOTAV3.

This module takes raw retrieval results plus high‑level memory and CAG
information and produces a `ContextPayload` structure that the persona
layer can flatten into text for the LLM.
"""

from __future__ import annotations

from typing import List, Optional

from IOTAV3.backend.context_engine.context_payload import ContextPayload, Passage

__all__ = ["ContextPayload", "build_context_payload"]


def _trim_snippet(content: str, max_chars: int = 1800) -> str:
    if len(content) <= max_chars:
        return content
    return content[: max_chars - 3] + "..."


def build_context_payload(
    *,
    query: str,
    chunks: List[dict],
    session_summary: Optional[str],
    user_profile_summary: Optional[str],
    cag_block: str,
) -> ContextPayload:
    """
    Build a ContextPayload from retrieved chunks and high‑level memory.

    The `chunks` list is expected to contain dictionaries with at least:
    - document_name
    - page_start
    - page_end
    - content or snippet (text for the passage)
    - similarity or cosine_similarity (float)
    """

    # Sort primarily by similarity descending; fall back to document
    # name to keep ordering stable. Accept both cosine_similarity and
    # similarity from the RPC.
    sorted_chunks = sorted(
        chunks,
        key=lambda row: (
            float(row.get("cosine_similarity") or row.get("similarity") or 0.0),
            str(row.get("document_name") or ""),
        ),
        reverse=True,
    )

    passages: List[Passage] = []
    for row in sorted_chunks:
        doc = str(row.get("document_name") or "").strip()
        if not doc:
            doc = "Unknown document"
        page_start = int(row.get("page_start") or 0)
        page_end = int(row.get("page_end") or page_start)
        content = str(row.get("snippet") or row.get("content") or "").strip()
        snippet = _trim_snippet(content) if content else ""
        similarity = float(row.get("cosine_similarity") or row.get("similarity") or 0.0)

        passages.append(
            Passage(
                document_name=doc,
                page_start=page_start,
                page_end=page_end,
                snippet=snippet,
                similarity=similarity,
            )
        )

    return ContextPayload(
        passages=passages,
        session_summary=session_summary,
        user_profile_summary=user_profile_summary,
        cag_block=cag_block,
    )


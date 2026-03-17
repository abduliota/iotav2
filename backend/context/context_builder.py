"""Structured context builder for the RAG pipeline.

This module provides a light-weight context object that can be logged,
inspected, and (when needed) rendered into human-readable form. The existing
`simple_rag.build_context` function still produces the main regulatory
context string for the LLM; this builder sits alongside it to give structure
to the rest of the inputs (memory, routing, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class ContextPayload:
    """High-level, structured view of the inputs to the LLM."""

    query: str
    intent: str
    in_scope: bool
    route: str
    documents: List[Dict[str, Any]]
    memory_items: List[Dict[str, Any]]
    profile: Dict[str, Any]
    session_summary: Dict[str, Any]
    cag_block: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_context_payload(
    *,
    query: str,
    intent: str,
    in_scope: bool,
    route: str,
    chunks: List[Dict[str, Any]],
    memory_items: List[Dict[str, Any]],
    profile: Optional[Dict[str, Any]],
    session_summary: Optional[Dict[str, Any]],
    cag_block: str,
) -> ContextPayload:
    """Assemble a structured context payload from raw components.

    This intentionally keeps only lightweight views of documents and memory
    (names, pages, ids, similarity), not full text.
    """

    docs_compact: List[Dict[str, Any]] = []
    for row in chunks:
        docs_compact.append(
            {
                "document_name": row.get("document_name") or "",
                "page_start": row.get("page_start"),
                "page_end": row.get("page_end"),
                "article_id": row.get("article_id")
                or row.get("article_number")
                or row.get("article"),
                "similarity": float(
                    row.get("cosine_similarity")
                    or row.get("similarity")
                    or row.get("score")
                    or 0.0
                ),
            }
        )

    mem_compact: List[Dict[str, Any]] = []
    for item in memory_items:
        mem_compact.append(
            {
                "id": item.get("memory_item_id") or item.get("id"),
                "type": item.get("type", ""),
                "similarity": float(item.get("similarity") or 0.0),
            }
        )

    return ContextPayload(
        query=query.strip(),
        intent=intent,
        in_scope=in_scope,
        route=route,
        documents=docs_compact,
        memory_items=mem_compact,
        profile=profile or {},
        session_summary=session_summary or {},
        cag_block=cag_block or "",
    )


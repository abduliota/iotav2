"""Supabase client for Stage 3: insert documents and chunks with embeddings."""
from __future__ import annotations

import os
from typing import Any
from uuid import UUID

from supabase import Client, create_client


_client: "Client | None" = None


def get_client() -> "Client":
    """Lazy-load Supabase client. Requires SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY env vars."""
    global _client
    if _client is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

        if not url:
            raise ValueError("SUPABASE_URL environment variable not set")
        if not key:
            raise ValueError("SUPABASE_SERVICE_ROLE_KEY environment variable not set")

        _client = create_client(url, key)
    return _client


def upsert_document(
    document_id: str,
    document_name: str,
    source_type: str | None = None,
    total_pages: int | None = None,
    document_code: str | None = None,
) -> UUID:
    """
    Upsert a document in the documents table.
    Returns the UUID (from existing row or newly created).
    """
    client = get_client()
    data: dict[str, Any] = {
        "id": document_id,
        "document_name": document_name,
    }
    if source_type:
        data["source_type"] = source_type
    if total_pages is not None:
        data["total_pages"] = total_pages
    if document_code:
        data["document_code"] = document_code

    result = (
        client.table("documents")
        .upsert(data, on_conflict="id")
        .execute()
    )
    if result.data:
        return UUID(result.data[0]["id"])
    return UUID(document_id)


def insert_chunks_batch(chunks: list[dict[str, Any]]) -> None:
    """
    Insert a batch of chunks into sama_nora_chunks.
    Each chunk dict must have: document_id, document_name, page_start, page_end,
    section_title (can be None), content, embedding (list[float]), token_count, language.
    """
    if not chunks:
        return
    client = get_client()
    client.table("sama_nora_chunks").insert(chunks).execute()


def get_document_by_id(document_id: str) -> dict[str, Any] | None:
    """Get document by ID. Returns None if not found."""
    client = get_client()
    result = client.table("documents").select("*").eq("id", document_id).execute()
    if result.data:
        return result.data[0]
    return None

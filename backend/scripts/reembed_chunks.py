"""
Re-embed all chunks in Supabase using the current embedding model (config: USE_MULTILINGUAL_EMBEDDING).
Run from backend: python scripts/reembed_chunks.py
Ensures query and DB embeddings use the same dimensions (e.g. 384 for multilingual-e5-small).
"""
from __future__ import annotations

import os
import sys

# Run from backend so config and imports resolve
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
os.chdir(BACKEND_DIR)

from config import CHUNK_BATCH_SIZE, EMBEDDING_DIMENSION
from embeddings import embed_chunk
from supabase_client import get_client


PAGE_SIZE = 1000  # Supabase default max; fetch in pages to get all rows


def main() -> None:
    client = get_client()
    # Fetch all rows in pages (Supabase returns max 1000 per request by default)
    rows: list[dict] = []
    start = 0
    while True:
        end = start + PAGE_SIZE - 1
        result = (
            client.table("sama_nora_chunks")
            .select("id, content")
            .range(start, end)
            .execute()
        )
        page = result.data or []
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            break
        start += PAGE_SIZE
        print(f"Fetched {len(rows)} rows so far...")
    total = len(rows)
    if total == 0:
        print("No chunks found in sama_nora_chunks.")
        return
    print(f"Re-embedding {total} chunks (embedding dimension={EMBEDDING_DIMENSION})...")
    updated = 0
    for i in range(0, total, CHUNK_BATCH_SIZE):
        batch = rows[i : i + CHUNK_BATCH_SIZE]
        for row in batch:
            chunk_id = row.get("id")
            content = row.get("content") or row.get("text") or ""
            if not content.strip():
                continue
            try:
                embedding = embed_chunk(content)
                if len(embedding) != EMBEDDING_DIMENSION:
                    print(f"Skip id={chunk_id}: wrong dimension {len(embedding)}")
                    continue
                client.table("sama_nora_chunks").update({"embedding": embedding}).eq("id", chunk_id).execute()
                updated += 1
                print(f"Updated id={chunk_id} ({updated}/{total})")
            except Exception as e:
                print(f"Error id={chunk_id}: {e}")
        print(f"Progress: {min(i + CHUNK_BATCH_SIZE, total)}/{total} processed, {updated} updated.")
    print(f"Done. Updated {updated} chunk embeddings.")


if __name__ == "__main__":
    main()

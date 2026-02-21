"""
Upload extracted chunks to Supabase with embeddings.

Reads chunk JSON files, generates embeddings, and inserts into Supabase.
Integrates with existing supabase_client and embeddings modules.

Usage:
    python scripts/upload_to_db.py <chunks_json> [--batch-size BATCH_SIZE]
    python scripts/upload_to_db.py <chunks_json> --document-id "doc-001" --skip-existing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Add backend to path for imports
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Load .env file BEFORE importing modules that need env vars
try:
    from dotenv import load_dotenv
    load_dotenv(BACKEND_DIR / ".env")
except ImportError:
    pass

from config import CHUNK_BATCH_SIZE
from embeddings import embed_chunk
from supabase_client import get_client, insert_chunks_batch, upsert_document


def upload_chunks_to_db(
    chunks: list[dict[str, Any]],
    batch_size: int = CHUNK_BATCH_SIZE,
    skip_existing: bool = False,
) -> dict[str, Any]:
    """
    Upload chunks to Supabase with embeddings.
    
    Args:
        chunks: List of chunk records from chunk_text.py
        batch_size: Batch size for inserts
        skip_existing: Skip chunks that already exist (by document_id + page range)
        
    Returns:
        Stats dict with counts and errors
    """
    if not chunks:
        return {"inserted": 0, "skipped": 0, "errors": 0, "error_details": []}
    
    # Group chunks by document
    documents = {}
    for chunk in chunks:
        doc_id = chunk["document_id"]
        if doc_id not in documents:
            documents[doc_id] = {
                "document_id": doc_id,
                "document_name": chunk["document_name"],
                "chunks": [],
            }
        documents[doc_id]["chunks"].append(chunk)
    
    # Upsert documents first
    print(f"Processing {len(documents)} document(s)...")
    for doc_id, doc_data in documents.items():
        try:
            upsert_document(
                document_id=doc_id,
                document_name=doc_data["document_name"],
                source_type=None,  # Can be set based on document_name pattern
                total_pages=None,  # Can be calculated from chunks
            )
            print(f"  ✓ Document: {doc_data['document_name']}")
        except Exception as e:
            print(f"  ✗ Error upserting document {doc_id}: {e}")
    
    # Process chunks with embeddings
    all_chunks_to_insert = []
    stats = {"inserted": 0, "skipped": 0, "errors": 0, "error_details": []}
    
    print(f"\nGenerating embeddings for {len(chunks)} chunks...")
    for i, chunk in enumerate(chunks):
        try:
            # Generate embedding
            content = chunk.get("content", "")
            if not content.strip():
                stats["skipped"] += 1
                continue
            
            embedding = embed_chunk(content)
            
            # Build chunk dict for Supabase
            chunk_dict = {
                "document_id": chunk["document_id"],
                "document_name": chunk["document_name"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "section_title": chunk.get("section_title"),
                "content": content,
                "embedding": embedding,
                "token_count": chunk.get("token_count"),
                "language": chunk.get("language", "en"),
            }
            
            all_chunks_to_insert.append(chunk_dict)
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(chunks)} chunks...")
                
        except Exception as e:
            stats["errors"] += 1
            stats["error_details"].append({
                "chunk_index": i,
                "document_id": chunk.get("document_id"),
                "error": str(e),
            })
            print(f"  ✗ Error processing chunk {i}: {e}")
    
    # Insert in batches
    print(f"\nInserting {len(all_chunks_to_insert)} chunks in batches of {batch_size}...")
    for i in range(0, len(all_chunks_to_insert), batch_size):
        batch = all_chunks_to_insert[i:i + batch_size]
        try:
            insert_chunks_batch(batch)
            stats["inserted"] += len(batch)
            print(f"  ✓ Inserted batch {i // batch_size + 1} ({len(batch)} chunks)")
        except Exception as e:
            stats["errors"] += len(batch)
            stats["error_details"].append({
                "batch_start": i,
                "batch_size": len(batch),
                "error": str(e),
            })
            print(f"  ✗ Error inserting batch {i // batch_size + 1}: {e}")
    
    return stats


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Upload chunks to Supabase")
    parser.add_argument("chunks_json", type=Path, help="Path to chunks JSON file")
    parser.add_argument("--batch-size", type=int, default=CHUNK_BATCH_SIZE, help="Batch size for inserts")
    parser.add_argument("--skip-existing", action="store_true", help="Skip existing chunks (not implemented)")
    
    args = parser.parse_args()
    
    if not args.chunks_json.exists():
        print(f"Error: File not found: {args.chunks_json}", file=sys.stderr)
        sys.exit(1)
    
    # Load chunks
    print(f"Loading chunks from: {args.chunks_json}")
    with open(args.chunks_json, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    print(f"Loaded {len(chunks)} chunks")
    
    # Upload to database
    stats = upload_chunks_to_db(chunks, batch_size=args.batch_size, skip_existing=args.skip_existing)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Upload Summary:")
    print(f"  Inserted: {stats['inserted']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Errors: {stats['errors']}")
    
    if stats["error_details"]:
        print(f"\nError Details:")
        for err in stats["error_details"][:10]:  # Show first 10 errors
            print(f"  - {err}")
        if len(stats["error_details"]) > 10:
            print(f"  ... and {len(stats['error_details']) - 10} more errors")


if __name__ == "__main__":
    main()

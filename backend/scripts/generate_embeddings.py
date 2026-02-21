"""
Generate embeddings for extracted chunks (standalone script).

Reads chunk JSON files and generates embeddings, optionally saving to new JSON.
Can be used independently or as part of the upload pipeline.

Usage:
    python scripts/generate_embeddings.py <chunks_json> [--output OUTPUT_JSON]
    python scripts/generate_embeddings.py <chunks_json> --output chunks_with_embeddings.json
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

from embeddings import embed_chunk


def generate_embeddings_for_chunks(
    chunks: list[dict[str, Any]],
    show_progress: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Generate embeddings for chunks.
    
    Args:
        chunks: List of chunk records (must have 'content' field)
        show_progress: Print progress messages
        
    Returns:
        Tuple of (chunks_with_embeddings, stats)
    """
    chunks_with_embeddings = []
    stats = {
        "total": len(chunks),
        "processed": 0,
        "skipped": 0,
        "errors": 0,
        "error_details": [],
    }
    
    for i, chunk in enumerate(chunks):
        try:
            content = chunk.get("content", "")
            if not content.strip():
                stats["skipped"] += 1
                if show_progress:
                    print(f"  Skipping chunk {i + 1} (empty content)")
                continue
            
            # Generate embedding
            embedding = embed_chunk(content)
            
            # Add embedding to chunk
            chunk_with_embedding = chunk.copy()
            chunk_with_embedding["embedding"] = embedding
            chunk_with_embedding["embedding_dimension"] = len(embedding)
            
            chunks_with_embeddings.append(chunk_with_embedding)
            stats["processed"] += 1
            
            if show_progress and (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(chunks)} chunks...")
                
        except Exception as e:
            stats["errors"] += 1
            stats["error_details"].append({
                "chunk_index": i,
                "document_id": chunk.get("document_id"),
                "error": str(e),
            })
            if show_progress:
                print(f"  ✗ Error processing chunk {i + 1}: {e}")
    
    return chunks_with_embeddings, stats


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate embeddings for chunks")
    parser.add_argument("chunks_json", type=Path, help="Path to chunks JSON file")
    parser.add_argument("--output", type=Path, help="Output JSON file (default: adds '_with_embeddings' suffix)")
    parser.add_argument("--no-progress", action="store_true", help="Don't show progress messages")
    
    args = parser.parse_args()
    
    if not args.chunks_json.exists():
        print(f"Error: File not found: {args.chunks_json}", file=sys.stderr)
        sys.exit(1)
    
    # Load chunks
    print(f"Loading chunks from: {args.chunks_json}")
    with open(args.chunks_json, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    print(f"Loaded {len(chunks)} chunks")
    
    # Generate embeddings
    print("\nGenerating embeddings...")
    chunks_with_embeddings, stats = generate_embeddings_for_chunks(
        chunks,
        show_progress=not args.no_progress,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("Embedding Generation Summary:")
    print(f"  Total chunks: {stats['total']}")
    print(f"  Processed: {stats['processed']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Errors: {stats['errors']}")
    
    if stats["error_details"]:
        print(f"\nError Details:")
        for err in stats["error_details"][:10]:  # Show first 10 errors
            print(f"  - {err}")
        if len(stats["error_details"]) > 10:
            print(f"  ... and {len(stats['error_details']) - 10} more errors")
    
    # Save output
    if args.output:
        output_file = args.output
    else:
        output_file = args.chunks_json.parent / f"{args.chunks_json.stem}_with_embeddings.json"
    
    print(f"\nSaving chunks with embeddings to: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks_with_embeddings, f, indent=2, ensure_ascii=False)
    
    print("Done!")


if __name__ == "__main__":
    main()

"""
Batch process PDFs: Extract → Chunk → Embed → Upload to Supabase.

Processes all PDFs in a directory through the complete pipeline.

Usage:
    python scripts/process_pdfs_batch.py backend/pdfs/nca
    python scripts/process_pdfs_batch.py backend/pdfs/nca --skip-existing
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import uuid
from pathlib import Path
from typing import Any

# Add backend to path for imports
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Load .env file BEFORE importing config
try:
    from dotenv import load_dotenv
    load_dotenv(BACKEND_DIR / ".env")
except ImportError:
    pass

from config import CHUNK_BATCH_SIZE, CHUNKS_OUTPUT_DIR, OUTPUT_DIR

# Import processing functions
# Import as modules since they're in the same directory
import importlib.util

def _load_module(name):
    """Load a module from the scripts directory."""
    script_path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

extract_text_module = _load_module("extract_text")
chunk_text_module = _load_module("chunk_text")
upload_to_db_module = _load_module("upload_to_db")

extract_text_from_pdf = extract_text_module.extract_text_from_pdf
chunk_text = chunk_text_module.chunk_text
upload_chunks_to_db = upload_to_db_module.upload_chunks_to_db


def process_pdf_to_supabase(
    pdf_path: Path,
    document_id: str | None = None,
    document_name: str | None = None,
    output_dir: Path = OUTPUT_DIR,
    chunks_dir: Path = CHUNKS_OUTPUT_DIR,
    batch_size: int = CHUNK_BATCH_SIZE,
    skip_existing: bool = False,
) -> dict[str, Any]:
    """
    Process a single PDF through the complete pipeline.
    
    Returns:
        Stats dict with processing results
    """
    print(f"\n{'='*60}")
    print(f"Processing: {pdf_path.name}")
    print(f"{'='*60}")
    
    stats = {
        "pdf_path": str(pdf_path),
        "document_id": document_id,
        "document_name": document_name,
        "extraction": {},
        "chunking": {},
        "upload": {},
        "success": False,
    }
    
    # Step 1: Extract text
    print("\n[1/4] Extracting text from PDF...")
    try:
        pages, extraction_stats = extract_text_from_pdf(
            pdf_path,
            document_id=document_id,
            document_name=document_name,
        )
        stats["extraction"] = extraction_stats
        stats["document_id"] = extraction_stats["document_id"]
        stats["document_name"] = extraction_stats["document_name"]
        print(f"  ✓ Extracted {len(pages)} pages")
        print(f"  ✓ Native: {extraction_stats['pages_native']}, OCR: {extraction_stats['pages_ocr']}")
    except Exception as e:
        print(f"  ✗ Extraction failed: {e}")
        stats["extraction"]["error"] = str(e)
        return stats
    
    # Step 2: Chunk text
    print("\n[2/4] Chunking text...")
    try:
        chunks = chunk_text(pages)
        stats["chunking"] = {
            "total_chunks": len(chunks),
            "total_pages": len(pages),
        }
        print(f"  ✓ Created {len(chunks)} chunks from {len(pages)} pages")
    except Exception as e:
        print(f"  ✗ Chunking failed: {e}")
        stats["chunking"]["error"] = str(e)
        return stats
    
    # Step 3: Upload to Supabase (includes embedding generation)
    print("\n[3/4] Generating embeddings and uploading to Supabase...")
    try:
        upload_stats = upload_chunks_to_db(
            chunks,
            batch_size=batch_size,
            skip_existing=skip_existing,
        )
        stats["upload"] = upload_stats
        print(f"  ✓ Uploaded: {upload_stats['inserted']} chunks")
        if upload_stats["errors"] > 0:
            print(f"  ⚠ Errors: {upload_stats['errors']}")
    except Exception as e:
        print(f"  ✗ Upload failed: {e}")
        stats["upload"]["error"] = str(e)
        return stats
    
    stats["success"] = True
    print(f"\n✓ Successfully processed: {pdf_path.name}")
    
    return stats


def process_directory(
    pdf_dir: Path,
    output_dir: Path = OUTPUT_DIR,
    chunks_dir: Path = CHUNKS_OUTPUT_DIR,
    batch_size: int = CHUNK_BATCH_SIZE,
    skip_existing: bool = False,
    pattern: str = "*.pdf",
) -> dict[str, Any]:
    """
    Process all PDFs in a directory.
    
    Returns:
        Summary stats
    """
    if not pdf_dir.exists():
        raise FileNotFoundError(f"Directory not found: {pdf_dir}")
    
    pdf_files = list(pdf_dir.glob(pattern))
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return {"total": 0, "processed": 0, "failed": 0, "results": []}
    
    print(f"\nFound {len(pdf_files)} PDF file(s) to process")
    print(f"Output directory: {output_dir}")
    print(f"Chunks directory: {chunks_dir}")
    
    results = []
    processed = 0
    failed = 0
    
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")
        
        # Generate document ID and name if not provided
        document_id = str(uuid.uuid4())
        document_name = pdf_path.stem.replace("_", " ").replace("-", " ").title()
        
        stats = process_pdf_to_supabase(
            pdf_path,
            document_id=document_id,
            document_name=document_name,
            output_dir=output_dir,
            chunks_dir=chunks_dir,
            batch_size=batch_size,
            skip_existing=skip_existing,
        )
        
        results.append(stats)
        if stats["success"]:
            processed += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total PDFs: {len(pdf_files)}")
    print(f"Successfully processed: {processed}")
    print(f"Failed: {failed}")
    
    if processed > 0:
        total_chunks = sum(r["upload"].get("inserted", 0) for r in results if r["success"])
        print(f"Total chunks uploaded: {total_chunks}")
    
    return {
        "total": len(pdf_files),
        "processed": processed,
        "failed": failed,
        "results": results,
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Batch process PDFs: Extract → Chunk → Embed → Upload"
    )
    parser.add_argument(
        "pdf_dir",
        type=Path,
        help="Directory containing PDF files (or path to single PDF)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for page JSON files",
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=CHUNKS_OUTPUT_DIR,
        help="Output directory for chunk JSON files",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=CHUNK_BATCH_SIZE,
        help="Batch size for database inserts",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip existing chunks (not fully implemented)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.pdf",
        help="File pattern to match (default: *.pdf)",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save processing results to JSON file",
    )
    
    args = parser.parse_args()
    
    # Ensure output directories exist
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.chunks_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if it's a single file or directory
    if args.pdf_dir.is_file() and args.pdf_dir.suffix.lower() == ".pdf":
        # Single PDF file
        document_id = str(uuid.uuid4())
        document_name = args.pdf_dir.stem.replace("_", " ").replace("-", " ").title()
        
        stats = process_pdf_to_supabase(
            args.pdf_dir,
            document_id=document_id,
            document_name=document_name,
            output_dir=args.output_dir,
            chunks_dir=args.chunks_dir,
            batch_size=args.batch_size,
            skip_existing=args.skip_existing,
        )
        
        if args.save_json:
            output_file = args.output_dir / f"{args.pdf_dir.stem}_processing.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
            print(f"\nSaved processing results to: {output_file}")
        
        sys.exit(0 if stats["success"] else 1)
    else:
        # Directory of PDFs
        try:
            summary = process_directory(
                args.pdf_dir,
                output_dir=args.output_dir,
                chunks_dir=args.chunks_dir,
                batch_size=args.batch_size,
                skip_existing=args.skip_existing,
                pattern=args.pattern,
            )
            
            if args.save_json:
                output_file = args.output_dir / "batch_processing_summary.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)
                print(f"\nSaved batch summary to: {output_file}")
            
            sys.exit(0 if summary["failed"] == 0 else 1)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()

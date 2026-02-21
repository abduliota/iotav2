"""
Text chunking logic with section detection and token-aware splitting.

Chunks extracted text into semantic units:
- Section detection (Article X, Section X, Arabic equivalents)
- Paragraph/sentence splitting
- Token-aware chunking with overlap
- Language detection (Arabic/English)
- Metadata extraction

Usage:
    python scripts/chunk_text.py <pages_json> [--output-dir OUTPUT_DIR]
    python scripts/chunk_text.py <pages_json> --chunk-size 500 --chunk-overlap 120
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# Add backend to path for imports
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv(BACKEND_DIR / ".env")
except ImportError:
    pass

from config import (
    BACKTRACK_TOKENS,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKS_OUTPUT_DIR,
    LONG_SECTION_TOKEN_THRESHOLD,
    QWEN_MODEL,
    SHORT_PAGE_TOKEN_THRESHOLD,
)

# Try to import tokenizer
try:
    from transformers import AutoTokenizer
    _tokenizer = None
    
    def get_tokenizer():
        """Lazy-load tokenizer."""
        global _tokenizer
        if _tokenizer is None:
            _tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
        return _tokenizer
except ImportError:
    # Fallback: simple word-based token counting
    def get_tokenizer():
        """Fallback tokenizer - word-based."""
        class SimpleTokenizer:
            def encode(self, text: str) -> list[int]:
                # Rough estimate: ~1.3 tokens per word
                words = text.split()
                return list(range(len(words)))  # Dummy tokens
            
            def decode(self, tokens: list[int]) -> str:
                return ""
        return SimpleTokenizer()


def detect_headings(text: str) -> list[tuple[int, str]]:
    """
    Detect section headings in text.
    Returns list of (line_index, heading_text) tuples.
    """
    headings = []
    lines = text.split("\n")
    
    # Patterns for headings
    patterns = [
        r"^(Article\s+\d+[:\s]?)",  # Article 1:
        r"^(Section\s+\d+[:\s]?)",   # Section 1:
        r"^(المادة\s+\d+[:\s]?)",   # Arabic: المادة
        r"^(الفصل\s+\d+[:\s]?)",    # Arabic: الفصل
        r"^(Chapter\s+\d+[:\s]?)",   # Chapter 1:
        r"^(Part\s+\d+[:\s]?)",     # Part 1:
    ]
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if len(line_stripped) < 3:
            continue
        
        # Check if line matches heading pattern and is short (likely a heading)
        for pattern in patterns:
            if re.match(pattern, line_stripped, re.IGNORECASE):
                if len(line_stripped) < 100:  # Headings are usually short
                    headings.append((i, line_stripped))
                    break
    
    return headings


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text."""
    try:
        return len(tokenizer.encode(text))
    except Exception:
        # Fallback: word count * 1.3
        return int(len(text.split()) * 1.3)


def split_by_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs (double newlines)."""
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def chunk_text(
    pages: list[dict[str, Any]],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """
    Chunk pages into semantic units.
    
    Args:
        pages: List of page records from extract_text.py
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens
        
    Returns:
        List of chunk records
        
    Chunk record format:
        {
            "document_id": str,
            "document_name": str,
            "page_start": int,
            "page_end": int,
            "section_title": str | None,
            "content": str,
            "token_count": int,
            "language": str,
        }
    """
    tokenizer = get_tokenizer()
    chunks = []
    
    # Group pages by document
    current_doc_id = None
    current_doc_name = None
    accumulated_text = ""
    accumulated_pages = []
    current_section_title = None
    
    for page in pages:
        doc_id = page["document_id"]
        doc_name = page["document_name"]
        page_num = page["page_number"]
        text = page["cleaned_text"]
        lang = page.get("language", "en")
        
        # New document - flush previous
        if current_doc_id and current_doc_id != doc_id:
            if accumulated_text:
                chunks.extend(_chunk_accumulated_text(
                    current_doc_id,
                    current_doc_name,
                    accumulated_text,
                    accumulated_pages,
                    current_section_title,
                    lang,
                    chunk_size,
                    chunk_overlap,
                    tokenizer,
                ))
            accumulated_text = ""
            accumulated_pages = []
            current_section_title = None
        
        current_doc_id = doc_id
        current_doc_name = doc_name
        
        # Detect section headings
        headings = detect_headings(text)
        if headings:
            # If we have accumulated text, chunk it before starting new section
            if accumulated_text:
                chunks.extend(_chunk_accumulated_text(
                    current_doc_id,
                    current_doc_name,
                    accumulated_text,
                    accumulated_pages,
                    current_section_title,
                    lang,
                    chunk_size,
                    chunk_overlap,
                    tokenizer,
                ))
                accumulated_text = ""
                accumulated_pages = []
            
            # Use first heading as section title
            current_section_title = headings[0][1]
        
        # Accumulate text
        if accumulated_text:
            accumulated_text += "\n\n"
        accumulated_text += text
        accumulated_pages.append(page_num)
        
        # Check if we should chunk now
        tokens = count_tokens(accumulated_text, tokenizer)
        
        # If too short, continue accumulating
        if tokens < SHORT_PAGE_TOKEN_THRESHOLD:
            continue
        
        # If too long, split by paragraphs first
        if tokens > LONG_SECTION_TOKEN_THRESHOLD:
            chunks.extend(_chunk_long_text(
                current_doc_id,
                current_doc_name,
                accumulated_text,
                accumulated_pages,
                current_section_title,
                lang,
                chunk_size,
                chunk_overlap,
                tokenizer,
            ))
            accumulated_text = ""
            accumulated_pages = []
            current_section_title = None
        elif tokens >= chunk_size:
            # Chunk accumulated text
            chunks.extend(_chunk_accumulated_text(
                current_doc_id,
                current_doc_name,
                accumulated_text,
                accumulated_pages,
                current_section_title,
                lang,
                chunk_size,
                chunk_overlap,
                tokenizer,
            ))
            # Keep overlap for next chunk
            overlap_text = _get_overlap_text(accumulated_text, chunk_overlap, tokenizer)
            accumulated_text = overlap_text + "\n\n" if overlap_text else ""
            accumulated_pages = accumulated_pages[-1:] if accumulated_pages else []
    
    # Flush remaining text
    if accumulated_text:
        chunks.extend(_chunk_accumulated_text(
            current_doc_id,
            current_doc_name,
            accumulated_text,
            accumulated_pages,
            current_section_title,
            lang if pages else "en",
            chunk_size,
            chunk_overlap,
            tokenizer,
        ))
    
    return chunks


def _chunk_accumulated_text(
    doc_id: str,
    doc_name: str,
    text: str,
    page_nums: list[int],
    section_title: str | None,
    lang: str,
    chunk_size: int,
    chunk_overlap: int,
    tokenizer,
) -> list[dict[str, Any]]:
    """Chunk accumulated text into size-appropriate chunks."""
    chunks = []
    tokens = count_tokens(text, tokenizer)
    
    if tokens <= chunk_size:
        # Single chunk
        chunks.append({
            "document_id": doc_id,
            "document_name": doc_name,
            "page_start": min(page_nums) if page_nums else 1,
            "page_end": max(page_nums) if page_nums else 1,
            "section_title": section_title,
            "content": text,
            "token_count": tokens,
            "language": lang,
        })
    else:
        # Split into multiple chunks
        sentences = re.split(r"([.!?؟؛]\s+)", text)
        current_chunk = ""
        current_tokens = 0
        chunk_start_page = min(page_nums) if page_nums else 1
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = count_tokens(sentence, tokenizer)
            
            if current_tokens + sentence_tokens > chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    "document_id": doc_id,
                    "document_name": doc_name,
                    "page_start": chunk_start_page,
                    "page_end": chunk_start_page,  # Approximate
                    "section_title": section_title,
                    "content": current_chunk.strip(),
                    "token_count": current_tokens,
                    "language": lang,
                })
                
                # Start new chunk with overlap
                overlap_text = _get_overlap_text(current_chunk, chunk_overlap, tokenizer)
                current_chunk = overlap_text + sentence if overlap_text else sentence
                current_tokens = count_tokens(current_chunk, tokenizer)
            else:
                current_chunk += sentence
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                "document_id": doc_id,
                "document_name": doc_name,
                "page_start": chunk_start_page,
                "page_end": max(page_nums) if page_nums else chunk_start_page,
                "section_title": section_title,
                "content": current_chunk.strip(),
                "token_count": current_tokens,
                "language": lang,
            })
    
    return chunks


def _chunk_long_text(
    doc_id: str,
    doc_name: str,
    text: str,
    page_nums: list[int],
    section_title: str | None,
    lang: str,
    chunk_size: int,
    chunk_overlap: int,
    tokenizer,
) -> list[dict[str, Any]]:
    """Chunk long text by splitting paragraphs first."""
    paragraphs = split_by_paragraphs(text)
    chunks = []
    current_chunk = ""
    current_tokens = 0
    chunk_start_page = min(page_nums) if page_nums else 1
    
    for para in paragraphs:
        para_tokens = count_tokens(para, tokenizer)
        
        if para_tokens > chunk_size:
            # Paragraph itself is too long - split by sentences
            if current_chunk:
                chunks.append({
                    "document_id": doc_id,
                    "document_name": doc_name,
                    "page_start": chunk_start_page,
                    "page_end": chunk_start_page,
                    "section_title": section_title,
                    "content": current_chunk.strip(),
                    "token_count": current_tokens,
                    "language": lang,
                })
                current_chunk = ""
                current_tokens = 0
            
            # Split paragraph
            para_chunks = _chunk_accumulated_text(
                doc_id, doc_name, para, [chunk_start_page],
                section_title, lang, chunk_size, chunk_overlap, tokenizer
            )
            chunks.extend(para_chunks)
        elif current_tokens + para_tokens > chunk_size:
            # Save current chunk
            chunks.append({
                "document_id": doc_id,
                "document_name": doc_name,
                "page_start": chunk_start_page,
                "page_end": chunk_start_page,
                "section_title": section_title,
                "content": current_chunk.strip(),
                "token_count": current_tokens,
                "language": lang,
            })
            
            # Start new chunk with overlap
            overlap_text = _get_overlap_text(current_chunk, chunk_overlap, tokenizer)
            current_chunk = overlap_text + "\n\n" + para if overlap_text else para
            current_tokens = count_tokens(current_chunk, tokenizer)
        else:
            if current_chunk:
                current_chunk += "\n\n"
            current_chunk += para
            current_tokens += para_tokens
    
    # Add final chunk
    if current_chunk.strip():
        chunks.append({
            "document_id": doc_id,
            "document_name": doc_name,
            "page_start": chunk_start_page,
            "page_end": max(page_nums) if page_nums else chunk_start_page,
            "section_title": section_title,
            "content": current_chunk.strip(),
            "token_count": current_tokens,
            "language": lang,
        })
    
    return chunks


def _get_overlap_text(text: str, overlap_tokens: int, tokenizer) -> str:
    """Get last N tokens of text for overlap."""
    if not text or overlap_tokens <= 0:
        return ""
    
    tokens = tokenizer.encode(text)
    if len(tokens) <= overlap_tokens:
        return text
    
    overlap_tokens_list = tokens[-overlap_tokens:]
    try:
        return tokenizer.decode(overlap_tokens_list)
    except Exception:
        # Fallback: last N words
        words = text.split()
        return " ".join(words[-overlap_tokens:])


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Chunk extracted text")
    parser.add_argument("pages_json", type=Path, help="Path to pages JSON file")
    parser.add_argument("--output-dir", type=Path, default=CHUNKS_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Chunk size in tokens")
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP, help="Overlap in tokens")
    
    args = parser.parse_args()
    
    if not args.pages_json.exists():
        print(f"Error: File not found: {args.pages_json}", file=sys.stderr)
        sys.exit(1)
    
    # Load pages
    with open(args.pages_json, "r", encoding="utf-8") as f:
        pages = json.load(f)
    
    # Chunk text
    chunks = chunk_text(pages, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    
    print(f"\nChunking complete:")
    print(f"  Input pages: {len(pages)}")
    print(f"  Output chunks: {len(chunks)}")
    print(f"  Average tokens per chunk: {sum(c['token_count'] for c in chunks) // len(chunks) if chunks else 0}")
    
    # Save chunks
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / f"{args.pages_json.stem}_chunks.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"\nSaved chunks to: {output_file}")


if __name__ == "__main__":
    main()

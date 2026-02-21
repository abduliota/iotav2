"""
PDF text extraction with PyMuPDF native extraction and OCR fallback.

Main extraction script for processing PDF documents:
- PyMuPDF native text extraction
- OCR fallback for scanned PDFs (PaddleOCR)
- Text cleaning (headers/footers, whitespace normalization)
- Scanned PDF detection
- Arabic OCR support

Usage:
    python scripts/extract_text.py <pdf_path> [--output-dir OUTPUT_DIR]
    python scripts/extract_text.py <pdf_path> --document-name "Document Name" --document-id "doc-001"
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

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
    HEADER_FOOTER_FREQUENCY_THRESHOLD,
    NATIVE_TEXT_MIN_LEN,
    OCR_DPI,
    OUTPUT_DIR,
    SENTENCE_END_CHARS,
    USE_OCR,
)

# Try to import OCR and cleaning modules
try:
    from ocr import ocr_image
except ImportError:
    # Fallback: create minimal OCR function if module doesn't exist
    def ocr_image(image_bytes: bytes) -> str:
        """Fallback OCR - returns empty if PaddleOCR not available."""
        try:
            from paddleocr import PaddleOCR
            import cv2
            import numpy as np
            
            ocr = PaddleOCR(use_gpu=True, use_angle_cls=True, lang="en", show_log=False)
            arr = np.frombuffer(image_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return ""
            result = ocr.ocr(img, cls=True)
            if result and result[0]:
                lines = []
                for line in result[0]:
                    if line and len(line) >= 2:
                        text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                        if text:
                            lines.append(text)
                return "\n".join(lines)
        except Exception:
            pass
        return ""

# Arabic Unicode range
ARABIC_START, ARABIC_END = 0x0600, 0x06FF


def detect_language(text: str) -> str:
    """Detect language: Arabic chars > 60% -> 'ar', else 'en'."""
    if not text.strip():
        return "en"
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return "en"
    arabic = sum(1 for c in chars if ARABIC_START <= ord(c) <= ARABIC_END)
    return "ar" if (arabic / len(chars)) > 0.6 else "en"


def normalize_whitespace(text: str) -> str:
    """Single spaces, collapse newlines, trim."""
    if not text:
        return ""
    import re
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fix_hyphenated_line_breaks(text: str) -> str:
    """Join 'word-\\nnext' into 'wordnext' (e.g. regula-\\ntion -> regulation)."""
    if not text:
        return ""
    import re
    return re.sub(r"-\s*\n\s*", "", text)


def merge_broken_sentences(text: str) -> str:
    """If a line does not end with sentence-ending chars, merge with next line."""
    if not text:
        return ""
    lines = [ln.rstrip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return ""
    end_chars = set(SENTENCE_END_CHARS)
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        while i + 1 < len(lines) and line and line[-1] not in end_chars:
            i += 1
            line = line + " " + lines[i].strip()
        out.append(line)
        i += 1
    return "\n".join(out)


def clean_page_text(text: str) -> str:
    """Run per-page cleaning only (no cross-page header/footer)."""
    if not text:
        return ""
    t = normalize_whitespace(text)
    t = fix_hyphenated_line_breaks(t)
    t = merge_broken_sentences(t)
    return normalize_whitespace(t)


def _first_lines(text: str, n: int = 3) -> list[str]:
    """First n non-empty lines."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    return lines[:n]


def _last_lines(text: str, n: int = 3) -> list[str]:
    """Last n non-empty lines."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    return lines[-n:] if len(lines) >= n else lines


def remove_repeated_headers_footers(page_texts: list[str]) -> list[str]:
    """
    Remove lines that appear in first/last 2-3 lines of more than threshold% of pages.
    Returns list of cleaned page texts (same order/length).
    """
    if not page_texts:
        return []
    n_pages = len(page_texts)
    threshold = max(1, int(n_pages * HEADER_FOOTER_FREQUENCY_THRESHOLD))
    to_remove = set()

    for get_lines in (_first_lines, _last_lines):
        counter = Counter()
        for text in page_texts:
            for line in get_lines(text, 3):
                if len(line) > 2 and not line.isdigit():
                    counter[line] += 1
        for line, count in counter.items():
            if count >= threshold:
                to_remove.add(line)

    def strip_lines(text: str) -> str:
        lines = text.split("\n")
        cleaned = [ln for ln in lines if ln.strip() not in to_remove]
        return "\n".join(cleaned)

    return [strip_lines(t) for t in page_texts]


def extract_text_from_pdf(
    pdf_path: Path,
    document_id: str | None = None,
    document_name: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Extract text from PDF with OCR fallback.
    
    Args:
        pdf_path: Path to PDF file
        document_id: Optional document ID (UUID string)
        document_name: Optional document name
        
    Returns:
        Tuple of (list of page records, stats dict)
        
    Page record format:
        {
            "document_id": str,
            "document_name": str,
            "page_number": int,
            "cleaned_text": str,
            "language": str,  # "en" or "ar"
            "extraction_method": str,  # "native" or "ocr"
        }
    """
    doc_id = document_id or str(uuid.uuid4())
    doc_name = document_name or pdf_path.stem.replace("_", " ").replace("-", " ").title()

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    raw_pages: list[str] = []
    pages_native = 0
    pages_ocr = 0
    extraction_methods: list[str] = []

    try:
        for i in range(len(doc)):
            page = doc[i]
            text = page.get_text("text") or ""
            
            # Use OCR if native text is too short
            if USE_OCR and len(text.strip()) <= NATIVE_TEXT_MIN_LEN:
                pix = page.get_pixmap(dpi=OCR_DPI, alpha=False)
                img_bytes = pix.tobytes("png")
                ocr_text = ocr_image(img_bytes)
                if ocr_text and len(ocr_text.strip()) > len(text.strip()):
                    text = ocr_text
                    pages_ocr += 1
                    extraction_methods.append("ocr")
                else:
                    pages_native += 1
                    extraction_methods.append("native")
            else:
                pages_native += 1
                extraction_methods.append("native")
            
            raw_pages.append(text)
    finally:
        doc.close()

    # Per-page cleaning
    cleaned_list = [clean_page_text(t) for t in raw_pages]
    
    # Cross-page header/footer removal
    cleaned_list = remove_repeated_headers_footers(cleaned_list)

    # Build page records
    records = []
    for i, cleaned in enumerate(cleaned_list):
        page_num = i + 1
        lang = detect_language(cleaned)
        records.append({
            "document_id": doc_id,
            "document_name": doc_name,
            "page_number": page_num,
            "cleaned_text": cleaned,
            "language": lang,
            "extraction_method": extraction_methods[i],
        })

    stats = {
        "pages_native": pages_native,
        "pages_ocr": pages_ocr,
        "total_pages": len(records),
        "document_id": doc_id,
        "document_name": doc_name,
    }
    
    return records, stats


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Extract text from PDF")
    parser.add_argument("pdf_path", type=Path, help="Path to PDF file")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory for JSON")
    parser.add_argument("--document-name", type=str, help="Document name")
    parser.add_argument("--document-id", type=str, help="Document ID (UUID)")
    parser.add_argument("--no-save", action="store_true", help="Don't save JSON, just print stats")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        records, stats = extract_text_from_pdf(
            args.pdf_path,
            document_id=args.document_id,
            document_name=args.document_name,
        )
        
        print(f"\nExtraction complete:")
        print(f"  Document: {stats['document_name']}")
        print(f"  Total pages: {stats['total_pages']}")
        print(f"  Native extraction: {stats['pages_native']}")
        print(f"  OCR extraction: {stats['pages_ocr']}")
        
        if not args.no_save:
            # Save page records as JSON
            output_file = args.output_dir / f"{args.pdf_path.stem}_pages.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            print(f"\nSaved page records to: {output_file}")
            
            # Save stats
            stats_file = args.output_dir / f"{args.pdf_path.stem}_stats.json"
            with open(stats_file, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
            print(f"Saved stats to: {stats_file}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

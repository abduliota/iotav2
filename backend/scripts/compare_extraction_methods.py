"""
Compare different PDF text extraction methods.

Compares:
- PyMuPDF native text extraction
- PyMuPDF OCR (PaddleOCR fallback)
- Performance metrics and reporting

Usage:
    python scripts/compare_extraction_methods.py <pdf_path> [--output-dir OUTPUT_DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

# Add backend to path for imports
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config import NATIVE_TEXT_MIN_LEN, OCR_DPI, OUTPUT_DIR, USE_OCR

# Try to import OCR
try:
    from ocr import ocr_image
except ImportError:
    def ocr_image(image_bytes: bytes) -> str:
        """Fallback OCR."""
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


def extract_native_text(pdf_path: Path) -> tuple[list[str], float]:
    """Extract text using PyMuPDF native extraction."""
    doc = fitz.open(pdf_path)
    texts = []
    start_time = time.time()
    
    try:
        for i in range(len(doc)):
            page = doc[i]
            text = page.get_text("text") or ""
            texts.append(text)
    finally:
        doc.close()
    
    elapsed = time.time() - start_time
    return texts, elapsed


def extract_with_ocr(pdf_path: Path) -> tuple[list[str], float]:
    """Extract text using OCR fallback for all pages."""
    doc = fitz.open(pdf_path)
    texts = []
    start_time = time.time()
    
    try:
        for i in range(len(doc)):
            page = doc[i]
            # Try native first
            text = page.get_text("text") or ""
            
            # If too short, use OCR
            if len(text.strip()) <= NATIVE_TEXT_MIN_LEN:
                pix = page.get_pixmap(dpi=OCR_DPI, alpha=False)
                img_bytes = pix.tobytes("png")
                ocr_text = ocr_image(img_bytes)
                if ocr_text and len(ocr_text.strip()) > len(text.strip()):
                    text = ocr_text
            texts.append(text)
    finally:
        doc.close()
    
    elapsed = time.time() - start_time
    return texts, elapsed


def extract_ocr_only(pdf_path: Path) -> tuple[list[str], float]:
    """Extract text using OCR for all pages (no native fallback)."""
    doc = fitz.open(pdf_path)
    texts = []
    start_time = time.time()
    
    try:
        for i in range(len(doc)):
            page = doc[i]
            pix = page.get_pixmap(dpi=OCR_DPI, alpha=False)
            img_bytes = pix.tobytes("png")
            text = ocr_image(img_bytes) or ""
            texts.append(text)
    finally:
        doc.close()
    
    elapsed = time.time() - start_time
    return texts, elapsed


def calculate_metrics(texts: list[str]) -> dict[str, Any]:
    """Calculate metrics for extracted text."""
    total_chars = sum(len(t) for t in texts)
    total_words = sum(len(t.split()) for t in texts)
    non_empty_pages = sum(1 for t in texts if t.strip())
    avg_chars_per_page = total_chars / len(texts) if texts else 0
    avg_words_per_page = total_words / len(texts) if texts else 0
    
    return {
        "total_pages": len(texts),
        "non_empty_pages": non_empty_pages,
        "total_characters": total_chars,
        "total_words": total_words,
        "avg_characters_per_page": round(avg_chars_per_page, 2),
        "avg_words_per_page": round(avg_words_per_page, 2),
        "empty_pages": len(texts) - non_empty_pages,
    }


def compare_methods(pdf_path: Path) -> dict[str, Any]:
    """Compare all extraction methods."""
    print(f"Comparing extraction methods for: {pdf_path.name}")
    print("=" * 60)
    
    results = {}
    
    # Method 1: Native text only
    print("\n1. Extracting with PyMuPDF native text...")
    native_texts, native_time = extract_native_text(pdf_path)
    native_metrics = calculate_metrics(native_texts)
    native_metrics["extraction_time_seconds"] = round(native_time, 2)
    results["native"] = {
        "texts": native_texts,
        "metrics": native_metrics,
    }
    print(f"   Time: {native_time:.2f}s")
    print(f"   Pages: {native_metrics['total_pages']}, Non-empty: {native_metrics['non_empty_pages']}")
    print(f"   Total chars: {native_metrics['total_characters']:,}")
    
    # Method 2: Native with OCR fallback
    if USE_OCR:
        print("\n2. Extracting with native + OCR fallback...")
        hybrid_texts, hybrid_time = extract_with_ocr(pdf_path)
        hybrid_metrics = calculate_metrics(hybrid_texts)
        hybrid_metrics["extraction_time_seconds"] = round(hybrid_time, 2)
        results["hybrid"] = {
            "texts": hybrid_texts,
            "metrics": hybrid_metrics,
        }
        print(f"   Time: {hybrid_time:.2f}s")
        print(f"   Pages: {hybrid_metrics['total_pages']}, Non-empty: {hybrid_metrics['non_empty_pages']}")
        print(f"   Total chars: {hybrid_metrics['total_characters']:,}")
        
        # Calculate improvement
        char_diff = hybrid_metrics["total_characters"] - native_metrics["total_characters"]
        char_improvement = (char_diff / native_metrics["total_characters"] * 100) if native_metrics["total_characters"] > 0 else 0
        print(f"   Improvement: +{char_diff:,} chars ({char_improvement:+.1f}%)")
    
    # Method 3: OCR only
    if USE_OCR:
        print("\n3. Extracting with OCR only...")
        ocr_texts, ocr_time = extract_ocr_only(pdf_path)
        ocr_metrics = calculate_metrics(ocr_texts)
        ocr_metrics["extraction_time_seconds"] = round(ocr_time, 2)
        results["ocr_only"] = {
            "texts": ocr_texts,
            "metrics": ocr_metrics,
        }
        print(f"   Time: {ocr_time:.2f}s")
        print(f"   Pages: {ocr_metrics['total_pages']}, Non-empty: {ocr_metrics['non_empty_pages']}")
        print(f"   Total chars: {ocr_metrics['total_characters']:,}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Fastest: Native ({native_time:.2f}s)")
    if USE_OCR:
        print(f"  Most text: Hybrid ({hybrid_metrics['total_characters']:,} chars)")
        print(f"  OCR overhead: {hybrid_time - native_time:.2f}s")
    
    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Compare PDF extraction methods")
    parser.add_argument("pdf_path", type=Path, help="Path to PDF file")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--save-texts", action="store_true", help="Save extracted texts to files")
    
    args = parser.parse_args()
    
    if not args.pdf_path.exists():
        print(f"Error: PDF not found: {args.pdf_path}", file=sys.stderr)
        sys.exit(1)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compare methods
    results = compare_methods(args.pdf_path)
    
    # Save comparison results
    output_file = args.output_dir / f"{args.pdf_path.stem}_comparison.json"
    
    # Prepare output (exclude full texts unless requested)
    output_data = {}
    for method, data in results.items():
        output_data[method] = {
            "metrics": data["metrics"],
        }
        if args.save_texts:
            output_data[method]["texts"] = data["texts"]
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved comparison results to: {output_file}")
    if args.save_texts:
        print("(Full texts included in JSON)")
    else:
        print("(Use --save-texts to include full extracted texts)")


if __name__ == "__main__":
    main()

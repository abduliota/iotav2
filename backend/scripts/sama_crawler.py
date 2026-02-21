"""
SAMA Rulebook crawler: Discover pages → download PDFs → extract → chunk → embed → upload to Supabase.

This script crawls `https://rulebook.sama.gov.sa/` starting from one or more section URLs,
collects PDF links (typically under `/sites/default/files/`), downloads the PDFs locally,
then runs the existing PDF pipeline (extract_text → chunk_text → embeddings → Supabase upload).

Usage:
    cd backend
    python scripts/sama_crawler.py

    # One specific start URL
    python scripts/sama_crawler.py --start-url "https://rulebook.sama.gov.sa/en/banking-sector-0"

    # Multiple sections by slug (will be converted to /en/<slug>)
    python scripts/sama_crawler.py --sections "banking-sector-0,finance-sector-0"

Outputs:
    - Downloads PDFs to: `backend/pdfs/sama/`
    - Writes metadata CSV to: `backend/sama_rulebook_pdfs.csv`
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("Error: Missing dependencies. Install with: pip install requests beautifulsoup4")
    sys.exit(1)

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

from config import CHUNK_BATCH_SIZE, CHUNKS_OUTPUT_DIR, OUTPUT_DIR

# Import processing functions
import importlib.util

def _load_module(name):
    """Load a module from the scripts directory."""
    script_path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

process_batch_module = _load_module("process_pdfs_batch")
process_pdf_to_supabase = process_batch_module.process_pdf_to_supabase


# Configuration
BASE_URL = "https://rulebook.sama.gov.sa"
DEFAULT_START_URLS = [
    "https://rulebook.sama.gov.sa/en/banking-sector-0",
    "https://rulebook.sama.gov.sa/en/finance-sector-0",
    "https://rulebook.sama.gov.sa/en/payment-systems-and-payment-services-providers",
    "https://rulebook.sama.gov.sa/en/money-exchange-sector-0",
    "https://rulebook.sama.gov.sa/en/credit-bureaus",
    "https://rulebook.sama.gov.sa/en/regulatory-sandbox",
]

REQUEST_TIMEOUT = 30
REQUEST_DELAY = 1.0  # Be polite

PDF_HOST = "rulebook.sama.gov.sa"
PDF_PATH_PREFIX = "/sites/default/files/"

# Storage (relative to backend/)
SAMA_PDF_DIR = BACKEND_DIR / "pdfs" / "sama"
SAMA_METADATA_CSV = BACKEND_DIR / "sama_rulebook_pdfs.csv"


def _headers() -> dict[str, str]:
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }


def fetch_html(url: str) -> str | None:
    try:
        resp = requests.get(url, headers=_headers(), timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"  ✗ Error fetching {url}: {e}")
        return None


def extract_page_title(html: str, fallback: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)
    if soup.title and soup.title.get_text(strip=True):
        return soup.title.get_text(strip=True)
    return fallback


def extract_nav_links(html: str, base_url: str) -> list[str]:
    """
    Extract links from the sidebar/book navigation and return absolute URLs.
    Focuses on the Drupal 'book' navigation blocks.
    """
    soup = BeautifulSoup(html, "html.parser")
    urls: list[str] = []

    # Sidebar navigation block (what you pasted in the message)
    sidebar = soup.find(id="block-rulebook-booknavigation")
    if sidebar:
        for a in sidebar.find_all("a", href=True):
            href = (a.get("href") or "").strip()
            if not href:
                continue
            if href.startswith("/"):
                full = urljoin(BASE_URL, href)
            else:
                full = urljoin(base_url, href)
            urls.append(full)

    # Additional book menus
    for nav in soup.find_all("nav", class_=re.compile(r"book-block-menu|book-menu", re.IGNORECASE)):
        for a in nav.find_all("a", href=True):
            href = (a.get("href") or "").strip()
            if not href:
                continue
            full = urljoin(BASE_URL, href) if href.startswith("/") else urljoin(base_url, href)
            urls.append(full)

    # Normalize + filter
    out: list[str] = []
    seen: set[str] = set()
    for u in urls:
        if "#" in u:
            u = u.split("#", 1)[0]
        if not u or u in seen:
            continue
        if "rulebook.sama.gov.sa" not in u:
            continue
        if "/en/" not in u:
            continue
        seen.add(u)
        out.append(u)
    return out


def extract_pdf_links(html: str, page_url: str) -> list[str]:
    """Extract PDF links from a page HTML and return absolute URLs."""
    soup = BeautifulSoup(html, "html.parser")
    pdfs: list[str] = []
    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        if ".pdf" not in href and PDF_PATH_PREFIX not in href:
            continue
        full = urljoin(page_url, href)
        parsed = urlparse(full)
        if parsed.netloc != PDF_HOST:
            continue
        # Most PDFs are served from /sites/default/files/
        if not parsed.path.startswith(PDF_PATH_PREFIX) and not parsed.path.lower().endswith(".pdf"):
            continue
        if full not in pdfs:
            pdfs.append(full)
    return pdfs


def _safe_filename(name: str) -> str:
    # Windows-friendly filename
    name = re.sub(r'[<>:"/\\\\|?*]+', "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:180] if len(name) > 180 else name


def pdf_filename_from_url(pdf_url: str) -> str:
    parsed = urlparse(pdf_url)
    base = Path(parsed.path).name
    if base.lower().endswith(".pdf") and base:
        return _safe_filename(base)
    # Fallback: stable hash
    h = hashlib.sha1(pdf_url.encode("utf-8")).hexdigest()[:16]
    return f"sama_{h}.pdf"


def download_pdf(pdf_url: str, dest_dir: Path, retries: int = 3) -> Path | None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = pdf_filename_from_url(pdf_url)
    dest_path = dest_dir / filename
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return dest_path

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(pdf_url, headers=_headers(), timeout=60, stream=True) as r:
                r.raise_for_status()
                with open(dest_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            return dest_path
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(2)
            continue
    print(f"  ✗ Failed downloading {pdf_url}: {last_err}")
    return None


def discover_book_pages(start_url: str) -> list[str]:
    """Discover all pages in the book navigation structure (bounded to /en/ on rulebook.sama.gov.sa)."""
    print(f"Discovering pages from: {start_url}")
    
    visited = set()
    to_visit = [start_url]
    all_pages = []
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    while to_visit:
        url = to_visit.pop(0)
        if url in visited:
            continue
        
        visited.add(url)
        print(f"  Checking: {url}")
        
        try:
            time.sleep(REQUEST_DELAY)
            html = fetch_html(url)
            if not html:
                continue
            
            # Add current page if it has content
            if url not in all_pages:
                all_pages.append(url)

            # Find links in the sidebar/book navigation; this is the safest to avoid exploding to unrelated pages.
            for full_url in extract_nav_links(html, url):
                if full_url not in visited and full_url not in to_visit:
                    to_visit.append(full_url)
        
        except Exception as e:
            print(f"  ✗ Error discovering from {url}: {e}")
            continue
    
    print(f"  Found {len(all_pages)} pages")
    return all_pages


def crawl_and_process_section(
    start_url: str,
    section_name: str | None = None,
    output_dir: Path = OUTPUT_DIR,
    chunks_dir: Path = CHUNKS_OUTPUT_DIR,
    batch_size: int = CHUNK_BATCH_SIZE,
) -> dict[str, Any]:
    """Crawl a section, download PDFs, then process PDFs through the existing pipeline."""
    print(f"\n{'='*60}")
    print(f"Processing section: {section_name or start_url}")
    print(f"{'='*60}")
    
    stats = {
        "section_url": start_url,
        "section_name": section_name,
        "discovery": {},
        "pdfs": {},
        "processing": {},
        "success": False,
    }
    
    # Step 1: Discover pages
    print("\n[1/4] Discovering pages...")
    try:
        pages_urls = discover_book_pages(start_url)
        stats["discovery"] = {
            "pages_found": len(pages_urls),
            "urls": pages_urls[:10],  # First 10 for reference
        }
        print(f"  ✓ Found {len(pages_urls)} pages")
    except Exception as e:
        print(f"  ✗ Discovery failed: {e}")
        stats["discovery"]["error"] = str(e)
        return stats
    
    # Step 2: Collect PDF links across discovered pages
    print(f"\n[2/4] Collecting PDF links from {len(pages_urls)} pages...")
    pdf_urls: list[str] = []
    pdf_to_sources: dict[str, list[dict[str, str]]] = {}
    for i, page_url in enumerate(pages_urls, 1):
        print(f"  [{i}/{len(pages_urls)}] {page_url}")
        html = fetch_html(page_url)
        if not html:
            continue
        title = extract_page_title(html, fallback=page_url)
        links = extract_pdf_links(html, page_url=page_url)
        for pdf_url in links:
            if pdf_url not in pdf_to_sources:
                pdf_urls.append(pdf_url)
                pdf_to_sources[pdf_url] = []
            pdf_to_sources[pdf_url].append({"page_url": page_url, "page_title": title})
        time.sleep(REQUEST_DELAY)

    stats["pdfs"]["pdf_links_found"] = len(pdf_urls)
    print(f"  ✓ Found {len(pdf_urls)} unique PDF link(s)")

    if not pdf_urls:
        stats["pdfs"]["error"] = "No PDF links found"
        return stats

    # Step 3: Download PDFs
    print(f"\n[3/4] Downloading PDFs to {SAMA_PDF_DIR} ...")
    downloaded: list[dict[str, str]] = []
    failed: list[str] = []
    SAMA_PDF_DIR.mkdir(parents=True, exist_ok=True)

    # Write/append metadata CSV
    csv_exists = SAMA_METADATA_CSV.exists()
    with open(SAMA_METADATA_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not csv_exists:
            w.writerow(["section_name", "section_url", "page_url", "page_title", "pdf_url", "filename"])
        for j, pdf_url in enumerate(pdf_urls, 1):
            print(f"  [{j}/{len(pdf_urls)}] {pdf_url}")
            path = download_pdf(pdf_url, SAMA_PDF_DIR)
            if path is None:
                failed.append(pdf_url)
                continue
            downloaded.append({"pdf_url": pdf_url, "path": str(path)})
            filename = path.name
            for src in pdf_to_sources.get(pdf_url, [])[:1]:
                # write one row per PDF (keep it small; we can extend later)
                w.writerow([
                    section_name or "",
                    start_url,
                    src.get("page_url", ""),
                    src.get("page_title", ""),
                    pdf_url,
                    filename,
                ])

    stats["pdfs"]["downloaded"] = len(downloaded)
    stats["pdfs"]["failed"] = len(failed)
    stats["pdfs"]["download_dir"] = str(SAMA_PDF_DIR)
    stats["pdfs"]["metadata_csv"] = str(SAMA_METADATA_CSV)

    if not downloaded:
        stats["pdfs"]["error"] = "All PDF downloads failed"
        return stats

    # Step 4: Process downloaded PDFs through the existing pipeline
    print("\n[4/4] Processing downloaded PDFs (extract → chunk → embed → upload)...")
    try:
        processed = 0
        proc_results: list[dict[str, Any]] = []
        for k, item in enumerate(downloaded, 1):
            pdf_path = Path(item["path"])
            pdf_url = item["pdf_url"]
            # Stable document_id derived from PDF URL so re-runs don't create duplicates.
            doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, pdf_url))
            doc_name = pdf_path.stem.replace("_", " ").replace("-", " ").strip()

            print(f"\n  [{k}/{len(downloaded)}] {pdf_path.name}")
            r = process_pdf_to_supabase(
                pdf_path=pdf_path,
                document_id=doc_id,
                document_name=doc_name,
                output_dir=OUTPUT_DIR,
                chunks_dir=CHUNKS_OUTPUT_DIR,
                batch_size=batch_size,
                skip_existing=False,
            )
            proc_results.append(r)
            if r.get("success"):
                processed += 1

        stats["processing"]["total_pdfs"] = len(downloaded)
        stats["processing"]["processed_ok"] = processed
        stats["processing"]["results_preview"] = proc_results[:3]
        print(f"\n  ✓ Processed {processed}/{len(downloaded)} PDFs successfully")
    except Exception as e:
        print(f"  ✗ Processing failed: {e}")
        stats["processing"]["error"] = str(e)
        return stats
    
    stats["success"] = True
    print(f"\n✓ Successfully processed section: {section_name or start_url}")
    
    return stats


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Crawl SAMA Rulebook and process through pipeline"
    )
    parser.add_argument(
        "--start-url",
        type=str,
        help="Starting URL (default: banking-sector-0)",
    )
    parser.add_argument(
        "--sections",
        type=str,
        help="Comma-separated list of section names (e.g., 'banking-sector-0,finance-sector-0')",
    )
    parser.add_argument(
        "--all-sections",
        action="store_true",
        help="Process all default sections",
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
    
    args = parser.parse_args()
    
    # Ensure output directories exist
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.chunks_dir.mkdir(parents=True, exist_ok=True)
    SAMA_PDF_DIR.mkdir(parents=True, exist_ok=True)
    
    # Determine which sections to process
    sections_to_process = []
    
    if args.all_sections:
        sections_to_process = [
            ("https://rulebook.sama.gov.sa/en/banking-sector-0", "Banking Sector"),
            ("https://rulebook.sama.gov.sa/en/finance-sector-0", "Finance Sector"),
            ("https://rulebook.sama.gov.sa/en/payment-systems-and-payment-services-providers", "Payment Systems"),
            ("https://rulebook.sama.gov.sa/en/money-exchange-sector-0", "Money Exchange Sector"),
            ("https://rulebook.sama.gov.sa/en/credit-bureaus", "Credit Bureaus"),
            ("https://rulebook.sama.gov.sa/en/regulatory-sandbox", "Regulatory Sandbox"),
        ]
    elif args.sections:
        section_names = [s.strip() for s in args.sections.split(",")]
        for name in section_names:
            url = f"https://rulebook.sama.gov.sa/en/{name}"
            sections_to_process.append((url, name.replace("-", " ").title()))
    elif args.start_url:
        sections_to_process = [(args.start_url, None)]
    else:
        # Default: just banking sector
        sections_to_process = [
            ("https://rulebook.sama.gov.sa/en/banking-sector-0", "Banking Sector")
        ]
    
    # Process each section
    all_results = []
    for url, name in sections_to_process:
        result = crawl_and_process_section(
            url,
            section_name=name,
            output_dir=args.output_dir,
            chunks_dir=args.chunks_dir,
            batch_size=args.batch_size,
        )
        all_results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total sections: {len(sections_to_process)}")
    successful = sum(1 for r in all_results if r["success"])
    print(f"Successfully processed: {successful}")
    print(f"Failed: {len(sections_to_process) - successful}")
    if successful > 0:
        total_pdfs = sum(int(r.get("processing", {}).get("processed_ok", 0)) for r in all_results if r.get("success"))
        print(f"Total PDFs processed OK: {total_pdfs}")
    
    # Save results
    results_file = args.output_dir / "sama_crawler_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to: {results_file}")


if __name__ == "__main__":
    main()

"""One-off NCA scraper: discover regulatory document PDFs and emit metadata CSV + downloads.

Usage (from project root):

    cd backend
    pip install -r requirements.txt
    playwright install chromium   # required for Option B fallback only
    python nca_scraper.py

Discovery uses HTML list pages first, then Option A (API/__NEXT_DATA__), then
Option B (Playwright) as fallback when fewer than 14 detail pages are found.
Playwright is optional; install with: playwright install chromium.

This script is intentionally polite:
- Sequential requests
- Small delay between requests
- Designed for one-time ingestion of all current NCA regulatory PDFs.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Optional, Set, Tuple

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse


BASE_URL = "https://nca.gov.sa"
START_URL = "https://nca.gov.sa/en/regulatory-documents/"
# The regulations listing is paginated client-side (page buttons without href),
# so we explicitly enumerate the known pages instead of trying to follow
# pagination links that do not exist in the HTML.
LIST_PAGES = [
    START_URL,
    f"{START_URL}?page=2",
    f"{START_URL}?page=3",
]

# Output locations (relative to backend directory)
BACKEND_DIR = Path(__file__).resolve().parent
CSV_PATH = BACKEND_DIR / "nca_documents.csv"
PDF_DIR = BACKEND_DIR / "pdfs" / "nca"

# Politeness settings
REQUEST_TIMEOUT = 20  # seconds
REQUEST_DELAY = 1.0  # seconds between HTTP requests

# Detail path patterns (Option A/B use these to recognize detail page URLs)
DETAIL_PATH_PATTERNS = (
    "/en/regulatory-documents/controls-list/",
    "/en/regulatory-documents/frameworks-and-standard-list/",
    "/en/regulatory-documents/guidelines-list/",
)


@dataclass
class DocumentRow:
    id: int
    source_url: str
    pdf_url: str
    filename: str
    page_title: str
    total_chunks: int
    created_at: str
    regulator: str
    domain: str


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr)


def fetch_html(url: str) -> Optional[str]:
    """Fetch a URL and return text, with basic error handling and delay."""
    _log(f"GET {url}")
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    except Exception as e:
        _log(f"ERROR fetching {url}: {e}")
        return None
    time.sleep(REQUEST_DELAY)
    if resp.status_code != 200:
        _log(f"ERROR {resp.status_code} for {url}")
        return None
    resp.encoding = resp.apparent_encoding or resp.encoding
    return resp.text


def _collect_detail_urls_from_json(obj: Any, base_url: str, found: Set[str]) -> None:
    """Recursively walk JSON and add any string that looks like a detail page URL."""
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_detail_urls_from_json(v, base_url, found)
    elif isinstance(obj, list):
        for v in obj:
            _collect_detail_urls_from_json(v, base_url, found)
    elif isinstance(obj, str):
        s = obj.strip()
        if not s:
            return
        for pattern in DETAIL_PATH_PATTERNS:
            if pattern in s:
                full = urljoin(base_url, s) if not s.startswith("http") else s
                parsed = urlparse(full)
                if parsed.netloc.endswith("nca.gov.sa"):
                    found.add(full)
                return


def fetch_detail_pages_from_api() -> Set[str]:
    """Option A: Discover detail page URLs from list page HTML (e.g. __NEXT_DATA__).

    Fetches each known list page, parses __NEXT_DATA__ if present, and collects
    any URLs matching regulatory document detail paths. Returns empty set on
    failure or if no such data is found.
    """
    detail_pages: Set[str] = set()
    for url in LIST_PAGES:
        html = fetch_html(url)
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        script = soup.find("script", id="__NEXT_DATA__", type="application/json")
        if not script or not script.string:
            continue
        try:
            data = json.loads(script.string)
        except json.JSONDecodeError:
            continue
        _collect_detail_urls_from_json(data, BASE_URL, detail_pages)
    return detail_pages


def fetch_detail_pages_with_playwright() -> Set[str]:
    """Option B (fallback): Use Playwright to render list pages and collect detail URLs.

    Opens the regulations listing, clicks pagination 2 and 3, and collects all
    links matching detail path patterns. Requires: pip install playwright &&
    playwright install chromium.
    """
    detail_pages: Set[str] = set()
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        _log("Playwright not installed; run: pip install playwright && playwright install chromium")
        return detail_pages

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(START_URL, wait_until="networkidle", timeout=REQUEST_TIMEOUT * 1000)
            time.sleep(1)

            def collect_links() -> None:
                for sel in (
                    "a[href*='/en/regulatory-documents/controls-list/']",
                    "a[href*='frameworks-and-standard-list/']",
                    "a[href*='guidelines-list/']",
                ):
                    for el in page.query_selector_all(sel):
                        href = el.get_attribute("href")
                        if href:
                            full = urljoin(START_URL, href)
                            parsed = urlparse(full)
                            if parsed.netloc.endswith("nca.gov.sa"):
                                detail_pages.add(full)

            collect_links()

            for page_num in ("2", "3"):
                for sel in (
                    f"button:has-text('{page_num}')",
                    f"a:has-text('{page_num}')",
                    f"[aria-label*='{page_num}']",
                ):
                    btn = page.query_selector(sel)
                    if btn:
                        try:
                            btn.click()
                            time.sleep(1.5)
                            collect_links()
                        except Exception:
                            pass
                        break
        finally:
            browser.close()
    return detail_pages


def discover_list_and_detail_links(start_url: str) -> Tuple[Set[str], Set[str]]:
    """Walk the regulations list pages and collect all detail pages.

    Returns (list_page_urls, detail_page_urls).
    """
    list_pages: Set[str] = set()
    detail_pages: Set[str] = set()

    # Explicitly visit all known list pages (1, 2, 3). The "pagination" UI on
    # the site is implemented client-side without href attributes, so we cannot
    # discover these URLs just by scanning <a href="...">.
    for url in LIST_PAGES:
        html = fetch_html(url)
        if not html:
            continue
        list_pages.add(url)

        soup = BeautifulSoup(html, "html.parser")

        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            full = urljoin(url, href)
            # Only stay within NCA domain
            parsed = urlparse(full)
            if not parsed.netloc.endswith("nca.gov.sa"):
                continue
            path = parsed.path.lower()

            # Detail pages: known path patterns under /en/regulatory-documents/
            if any(p in path for p in DETAIL_PATH_PATTERNS):
                detail_pages.add(full)

    return list_pages, detail_pages


def extract_pdfs_from_detail_page(detail_url: str) -> List[Tuple[str, str]]:
    """Return list of (pdf_url, filename) from a detail page."""
    html = fetch_html(detail_url)
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")

    # Page title: prefer main h1, fall back to <title>
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        page_title = h1.get_text(strip=True)
    else:
        if soup.title and soup.title.get_text(strip=True):
            page_title = soup.title.get_text(strip=True)
        else:
            page_title = detail_url

    pdfs: List[Tuple[str, str, str]] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href:
            continue
        full = urljoin(detail_url, href)
        parsed = urlparse(full)
        path_lower = parsed.path.lower()
        if not path_lower.endswith(".pdf"):
            continue
        filename = Path(parsed.path).name
        pdfs.append((full, filename, page_title))

    # Deduplicate by pdf_url
    seen: Set[str] = set()
    results: List[Tuple[str, str]] = []
    for pdf_url, filename, _title in pdfs:
        if pdf_url in seen:
            continue
        seen.add(pdf_url)
        results.append((pdf_url, filename))
    return results


def load_existing_csv(path: Path) -> Tuple[int, Set[Tuple[str, str]]]:
    """Return (max_id, existing_keys) where key is (pdf_url, filename)."""
    if not path.exists():
        return 9999, set()
    max_id = 9999
    existing: Set[Tuple[str, str]] = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row_id = int(row.get("id", "") or 0)
                if row_id > max_id:
                    max_id = row_id
            except ValueError:
                continue
            pdf_url = row.get("pdf_url") or ""
            filename = row.get("filename") or ""
            if pdf_url and filename:
                existing.add((pdf_url, filename))
    return max_id, existing


def append_rows_to_csv(path: Path, rows: Iterable[DocumentRow]) -> None:
    """Append new rows to CSV, creating it with header if needed."""
    fieldnames = [
        "id",
        "source_url",
        "pdf_url",
        "filename",
        "page_title",
        "total_chunks",
        "created_at",
        "regulator",
        "domain",
    ]
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row.id,
                    "source_url": row.source_url,
                    "pdf_url": row.pdf_url,
                    "filename": row.filename,
                    "page_title": row.page_title,
                    "total_chunks": row.total_chunks,
                    "created_at": row.created_at,
                    "regulator": row.regulator,
                    "domain": row.domain,
                }
            )


def download_pdf(pdf_url: str, dest: Path) -> bool:
    """Download a PDF to dest. Returns True on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        _log(f"SKIP existing file {dest}")
        return True

    _log(f"Downloading PDF {pdf_url} -> {dest}")
    try:
        with requests.get(pdf_url, stream=True, timeout=REQUEST_TIMEOUT) as r:
            if r.status_code != 200:
                _log(f"ERROR {r.status_code} for {pdf_url}")
                return False
            content_type = r.headers.get("Content-Type", "").lower()
            if "pdf" not in content_type and content_type:
                _log(f"WARNING unexpected Content-Type {content_type} for {pdf_url}")
            total_bytes = 0
            with dest.open("wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    f.write(chunk)
                    total_bytes += len(chunk)
        time.sleep(REQUEST_DELAY)
        if total_bytes == 0:
            _log(f"ERROR empty file for {pdf_url}")
            try:
                dest.unlink()
            except FileNotFoundError:
                pass
            return False
        return True
    except Exception as e:
        _log(f"ERROR downloading {pdf_url}: {e}")
        try:
            if dest.exists():
                dest.unlink()
        except Exception:
            pass
        return False


def run() -> None:
    _log("Starting NCA scraper")
    list_pages, detail_pages = discover_list_and_detail_links(START_URL)
    _log(f"Discovered {len(list_pages)} list pages and {len(detail_pages)} detail pages (HTML)")

    # Option A: try to get more detail URLs from API / __NEXT_DATA__
    try:
        api_urls = fetch_detail_pages_from_api()
        if api_urls:
            before = len(detail_pages)
            detail_pages |= api_urls
            _log(f"Option A (API/__NEXT_DATA__): added {len(detail_pages) - before} URLs; total detail pages {len(detail_pages)}")
    except Exception as e:
        _log(f"Option A failed: {e}")

    # Option B (fallback): use Playwright if we still have few detail pages
    if len(detail_pages) < 14:
        try:
            pw_urls = fetch_detail_pages_with_playwright()
            if pw_urls:
                before = len(detail_pages)
                detail_pages |= pw_urls
                _log(f"Option B (Playwright fallback): added {len(detail_pages) - before} URLs; total detail pages {len(detail_pages)}")
        except Exception as e:
            _log(f"Option B (Playwright) failed: {e}")

    max_id, existing_keys = load_existing_csv(CSV_PATH)
    _log(f"Loaded existing CSV {CSV_PATH} with max id={max_id} and {len(existing_keys)} existing entries")

    new_rows: List[DocumentRow] = []
    next_id = max_id + 1
    failed_downloads: List[Tuple[str, str, str]] = []  # (source_url, pdf_url, filename)

    for detail_url in sorted(detail_pages):
        html = fetch_html(detail_url)
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        h1 = soup.find("h1")
        if h1 and h1.get_text(strip=True):
            page_title = h1.get_text(strip=True)
        else:
            if soup.title and soup.title.get_text(strip=True):
                page_title = soup.title.get_text(strip=True)
            else:
                page_title = detail_url

        # Extract PDFs from this HTML we already fetched
        pdf_links: List[Tuple[str, str]] = []
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href:
                continue
            full = urljoin(detail_url, href)
            parsed = urlparse(full)
            if not parsed.path.lower().endswith(".pdf"):
                continue
            filename = Path(parsed.path).name
            pdf_links.append((full, filename))

        # Deduplicate PDFs within this page
        local_seen: Set[str] = set()
        final_links: List[Tuple[str, str]] = []
        for pdf_url, filename in pdf_links:
            key = (pdf_url, filename)
            if key in local_seen:
                continue
            local_seen.add(pdf_url)
            final_links.append((pdf_url, filename))

        if not final_links:
            _log(f"No PDFs found on {detail_url}")
            continue

        for pdf_url, filename in final_links:
            key = (pdf_url, filename)
            if key in existing_keys:
                _log(f"Already in CSV, skipping metadata for {pdf_url}")
                continue

            created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            row = DocumentRow(
                id=next_id,
                source_url=detail_url,
                pdf_url=pdf_url,
                filename=filename,
                page_title=page_title,
                total_chunks=0,
                created_at=created_at,
                regulator="NCA",
                domain="cybersecurity",
            )
            next_id += 1
            new_rows.append(row)

            # Download the PDF immediately for this row
            dest = PDF_DIR / filename
            if not download_pdf(pdf_url, dest):
                failed_downloads.append((detail_url, pdf_url, filename))

    if new_rows:
        _log(f"Appending {len(new_rows)} new rows to {CSV_PATH}")
        append_rows_to_csv(CSV_PATH, new_rows)
    else:
        _log("No new rows to append")

    if failed_downloads:
        failed_path = BACKEND_DIR / "nca_failed_downloads.csv"
        _log(f"Writing {len(failed_downloads)} failed downloads to {failed_path}")
        with failed_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["source_url", "pdf_url", "filename"])
            writer.writerows(failed_downloads)

    _log("NCA scraper finished")


if __name__ == "__main__":
    run()


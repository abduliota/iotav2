"""
Generic recursive site crawler for discovering PDF URLs.

BFS-crawls HTML pages within allowed domains, collects all .pdf links.
No content filtering. Supports checkpoint/resume and optional parallel workers.

Usage:
    from crawl_site import crawl_for_pdfs

    urls = crawl_for_pdfs(
        base_urls=["https://sdaia.gov.sa/en/"],
        allowed_domains=["sdaia.gov.sa"],
        max_pages=1500,
        max_depth=5,
        target_pdfs=500,
        output_path=Path("output/sdaia_discovered.json"),
        resume=False,
        workers=1,
    )
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import List, Set, Tuple
from urllib.parse import urljoin, urlparse

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    raise ImportError("Missing dependencies. Install with: pip install requests beautifulsoup4")

logger = logging.getLogger(__name__)

TIMEOUT = 60
MAX_RETRIES = 3
REQUEST_DELAY = 1.0  # seconds between requests


def _headers() -> dict:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }


def _normalize_domain(domain: str) -> str:
    """Strip www. and lowercase for comparison."""
    d = domain.lower().strip()
    if d.startswith("www."):
        d = d[4:]
    return d


def _is_allowed_domain(url: str, allowed_domains: List[str]) -> bool:
    parsed = urlparse(url)
    netloc = _normalize_domain(parsed.netloc or "")
    if not netloc:
        return False
    for allowed in allowed_domains:
        if _normalize_domain(allowed) in netloc or netloc.endswith("." + _normalize_domain(allowed)):
            return True
    return False


def _extract_links(html: str, page_url: str) -> Tuple[List[str], List[str]]:
    """Extract internal HTML links and PDF links from page. Returns (internal_links, pdf_links)."""
    soup = BeautifulSoup(html, "html.parser")
    internal: List[str] = []
    pdfs: List[str] = []
    seen_internal: Set[str] = set()
    seen_pdfs: Set[str] = set()

    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href or href.startswith("#") or href.startswith("javascript:"):
            continue
        full = urljoin(page_url, href)
        parsed = urlparse(full)
        path = (parsed.path or "").lower()
        if path.endswith(".pdf") or ".pdf?" in path:
            if full not in seen_pdfs:
                seen_pdfs.add(full)
                pdfs.append(full)
        else:
            if full not in seen_internal:
                seen_internal.add(full)
                internal.append(full)
    return internal, pdfs


def _fetch_html(url: str) -> str | None:
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(
                url,
                headers=_headers(),
                timeout=(10, TIMEOUT),
                allow_redirects=True,
            )
            resp.raise_for_status()
            ct = resp.headers.get("Content-Type", "").lower()
            if "text/html" not in ct and "application/xhtml" not in ct:
                return None
            return resp.text
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                time.sleep(2)
                continue
            logger.warning("Failed to fetch %s after %s retries: %s", url, MAX_RETRIES, last_exc)
            return None
    return None


def crawl_for_pdfs(
    base_urls: List[str],
    allowed_domains: List[str],
    max_pages: int = 2000,
    max_depth: int = 5,
    target_pdfs: int = 500,
    output_path: Path | None = None,
    resume: bool = False,
    workers: int = 1,
    request_delay: float = REQUEST_DELAY,
) -> List[str]:
    """
    BFS-crawl site for PDF URLs. No content filtering.

    Args:
        base_urls: Starting URLs for crawl.
        allowed_domains: Only follow links and collect PDFs from these domains.
        max_pages: Max HTML pages to fetch.
        max_depth: Max BFS depth.
        target_pdfs: Stop when we have at least this many PDFs (optional early exit).
        output_path: Save discovered URLs to this JSON file.
        resume: If True and output_path exists, load URLs from file instead of crawling.
        workers: Number of parallel fetch workers (1 = sequential).
        request_delay: Seconds to wait between requests.

    Returns:
        Sorted list of unique PDF URLs.
    """
    if resume and output_path and output_path.exists():
        try:
            data = json.loads(output_path.read_text(encoding="utf-8"))
            urls = data.get("urls", [])
            if urls:
                logger.info("Resuming: loaded %s PDF URLs from %s", len(urls), output_path)
                return sorted(urls)
        except Exception as e:
            logger.warning("Could not load checkpoint %s: %s. Starting fresh.", output_path, e)

    visited: Set[str] = set()
    pdf_urls: Set[str] = set()
    queue: deque[Tuple[str, int]] = deque()
    lock = threading.Lock()
    pages_fetched = 0
    last_request_time = [0.0]  # use list for mutability in closure

    def _rate_limit():
        elapsed = time.monotonic() - last_request_time[0]
        if elapsed < request_delay:
            time.sleep(request_delay - elapsed)
        last_request_time[0] = time.monotonic()

    for u in base_urls:
        parsed = urlparse(u)
        if not parsed.scheme or not parsed.netloc:
            continue
        if _is_allowed_domain(u, allowed_domains):
            queue.append((u, 0))

    def _process_page(url: str, depth: int) -> List[Tuple[str, int]]:
        nonlocal pages_fetched
        _rate_limit()
        html = _fetch_html(url)
        if not html:
            return []
        with lock:
            pages_fetched += 1
            if pages_fetched % 50 == 0:
                logger.info("Crawled %s pages, found %s PDFs so far", pages_fetched, len(pdf_urls))

        internal, pdfs = _extract_links(html, url)
        new_queue_items: List[Tuple[str, int]] = []
        next_depth = depth + 1
        if next_depth <= max_depth:
            for link in internal:
                if not _is_allowed_domain(link, allowed_domains):
                    continue
                with lock:
                    if link in visited:
                        continue
                    visited.add(link)
                new_queue_items.append((link, next_depth))
        with lock:
            for p in pdfs:
                if _is_allowed_domain(p, allowed_domains):
                    pdf_urls.add(p)
        return new_queue_items

    if workers <= 1:
        while queue and pages_fetched < max_pages and len(pdf_urls) < target_pdfs:
            url, depth = queue.popleft()
            if url in visited:
                continue
            visited.add(url)
            for item in _process_page(url, depth):
                queue.append(item)
    else:
        import concurrent.futures
        pending: Set[concurrent.futures.Future] = set()
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            while (queue or pending) and pages_fetched < max_pages and len(pdf_urls) < target_pdfs:
                while queue and len(pending) < workers * 2:
                    url, depth = queue.popleft()
                    with lock:
                        if url in visited:
                            continue
                        visited.add(url)
                    fut = executor.submit(_process_page, url, depth)
                    pending.add(fut)
                if not pending:
                    break
                done, pending = concurrent.futures.wait(
                    pending, timeout=1.0, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for fut in done:
                    try:
                        new_items = fut.result()
                        for item in new_items:
                            queue.append(item)
                    except Exception as e:
                        logger.warning("Worker error: %s", e)

    result = sorted(pdf_urls)
    logger.info("Crawl complete: %s pages, %s PDFs discovered", pages_fetched, len(result))

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "urls": result,
                    "crawled_pages": pages_fetched,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Saved discovered URLs to %s", output_path)

    return result

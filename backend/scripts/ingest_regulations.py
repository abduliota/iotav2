"""
Ingest regulations PDFs from SDAIA, NCA, SAMA, Aramco, and ISO into Supabase.

Workflow per source (except SAMA):
  1) Crawl site for PDF URLs (BFS, no filtering, target 500+ per source)
  2) Download PDFs into backend/pdfs/<source>/
  3) Process PDFs via process_pdfs_batch: extract → chunk → embed → upload

Usage (from backend/):

    # Ingest everything (SDAIA, NCA, Aramco, ISO - SAMA excluded from all)
    python scripts/ingest_regulations.py --source all

    # Single source
    python scripts/ingest_regulations.py --source sdaia
    python scripts/ingest_regulations.py --source nca
    python scripts/ingest_regulations.py --source sama
    python scripts/ingest_regulations.py --source aramco
    python scripts/ingest_regulations.py --source iso

    # Discovery only (no download/process)
    python scripts/ingest_regulations.py --source sdaia --discover-only

    # Resume from saved discovery
    python scripts/ingest_regulations.py --source sdaia --resume

    # Parallel crawl (faster discovery)
    python scripts/ingest_regulations.py --source sdaia --workers 3
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set
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

# Load .env file BEFORE importing config
try:
    from dotenv import load_dotenv

    load_dotenv(BACKEND_DIR / ".env")
except ImportError:
    pass

from config import CHUNK_BATCH_SIZE, CHUNKS_OUTPUT_DIR, OUTPUT_DIR


def _load_module(name: str) -> Any:
    """Load a module from the scripts directory by filename."""
    script_path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


process_pdfs_batch = _load_module("process_pdfs_batch")
sama_crawler = _load_module("sama_crawler")
crawl_site = _load_module("crawl_site")

DISCOVERY_OUTPUT_DIR = BACKEND_DIR / "output"
DISCOVERY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


TIMEOUT = 60  # total timeout budget per request (seconds)
MAX_RETRIES = 3
REQUEST_DELAY = 1.0  # delay between HTTP requests (seconds)
CURL_DELAY = 1.0  # delay between curl downloads (seconds)

LOG_DIR = BACKEND_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "ingest_regulations.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _headers() -> Dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,image/apng,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }


def fetch_html(url: str) -> str | None:
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("Fetching HTML (%s), attempt %s/%s", url, attempt, MAX_RETRIES)
            resp = requests.get(
                url,
                headers=_headers(),
                timeout=(10, TIMEOUT),
                allow_redirects=True,
            )
            resp.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return resp.text
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                logger.warning("Error fetching %s: %s (retrying)", url, exc)
                time.sleep(2)
                continue
            logger.error("Error fetching %s after retries: %s", url, last_exc)
            return None
    return None


def extract_pdfs_from_html(url: str) -> List[str]:
    html = fetch_html(url)
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    pdf_links: Set[str] = set()
    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        if not href.lower().endswith(".pdf"):
            continue
        full = urljoin(url, href)
        pdf_links.add(full)
    return sorted(pdf_links)


def safe_filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name or "file.pdf"
    # Basic sanitization for Windows file names.
    for ch in '<>:"/\\|?*':
        name = name.replace(ch, "_")
    return name or "file.pdf"


def download_with_requests(url: str, dest: Path, timeout: int = 60) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("SKIP (exists) %s (%s bytes)", dest.name, dest.stat().st_size)
        return True

    try:
        logger.info("Downloading via requests: %s -> %s", url, dest)
        with requests.get(
            url,
            headers=_headers(),
            timeout=(10, timeout),
            stream=True,
        ) as r:
            if r.status_code != 200:
                logger.error("FAIL %s -- status=%s", url, r.status_code)
                return False
            ct = r.headers.get("Content-Type", "").lower()
            if "pdf" not in ct:
                logger.warning("Unexpected Content-Type for %s: %s", url, ct)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
        time.sleep(REQUEST_DELAY)
        logger.info("OK downloaded %s (%s bytes)", dest.name, dest.stat().st_size)
        return True
    except requests.RequestException as exc:
        logger.error("FAIL %s -- error=%s", url, exc)
        return False


def download_with_curl(url: str, dest: Path, timeout: int = 120) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("SKIP (exists) %s (%s bytes)", dest.name, dest.stat().st_size)
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading via curl: %s -> %s", url, dest)
    cmd = ["curl", "-L", "--max-time", str(timeout), "-o", str(dest), url]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
        time.sleep(CURL_DELAY)
        logger.info("OK downloaded via curl %s (%s bytes)", dest.name, dest.stat().st_size)
        return True

    logger.error(
        "FAIL %s -- curl exit=%s, stderr=%s",
        url,
        result.returncode,
        result.stderr.strip(),
    )
    return False


# ---------------------------------------------------------------------------
# SDAIA (full crawl, target 500+ PDFs)
# ---------------------------------------------------------------------------

SDAIA_CRAWL_CONFIG = {
    "base_urls": [
        "https://sdaia.gov.sa/en/",
        "https://sdaia.gov.sa/ar/",
    ],
    "allowed_domains": ["sdaia.gov.sa"],
    "max_pages": 1500,
    "max_depth": 5,
    "target_pdfs": 500,
}


def discover_sdaia_pdfs(resume: bool = False, workers: int = 1) -> List[str]:
    logger.info("Crawling SDAIA for PDFs (target 500+)...")
    return crawl_site.crawl_for_pdfs(
        base_urls=SDAIA_CRAWL_CONFIG["base_urls"],
        allowed_domains=SDAIA_CRAWL_CONFIG["allowed_domains"],
        max_pages=SDAIA_CRAWL_CONFIG["max_pages"],
        max_depth=SDAIA_CRAWL_CONFIG["max_depth"],
        target_pdfs=SDAIA_CRAWL_CONFIG["target_pdfs"],
        output_path=DISCOVERY_OUTPUT_DIR / "ingest_sdaia_discovered.json",
        resume=resume,
        workers=workers,
        request_delay=REQUEST_DELAY,
    )


def ingest_sdaia(
    discover_only: bool = False,
    resume: bool = False,
    workers: int = 1,
) -> None:
    started_at = datetime.utcnow().isoformat() + "Z"
    stats: Dict[str, Any] = {
        "source": "sdaia",
        "started_at": started_at,
        "pdfs_discovered": 0,
        "pdfs_downloaded": 0,
        "pdfs_failed": 0,
        "pipeline_result": {},
    }

    pdf_urls = discover_sdaia_pdfs(resume=resume, workers=workers)
    stats["pdfs_discovered"] = len(pdf_urls)
    logger.info("Total SDAIA PDFs discovered: %s", len(pdf_urls))

    if discover_only:
        logger.info("Discovery only - skipping download and processing")
        return

    pdf_dir = BACKEND_DIR / "pdfs" / "sdaia"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    for url in pdf_urls:
        dest = pdf_dir / safe_filename_from_url(url)
        ok = download_with_requests(url, dest)
        if ok:
            stats["pdfs_downloaded"] += 1
        else:
            stats["pdfs_failed"] += 1

    logger.info("[PIPELINE] Processing SDAIA PDFs through RAG pipeline...")
    pipeline_result = process_pdfs_batch.process_directory(
        pdf_dir,
        output_dir=OUTPUT_DIR,
        chunks_dir=CHUNKS_OUTPUT_DIR,
        batch_size=CHUNK_BATCH_SIZE,
        skip_existing=False,
    )
    stats["pipeline_result"] = pipeline_result or {}
    stats["finished_at"] = datetime.utcnow().isoformat() + "Z"

    summary_dir = BACKEND_DIR / "output"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_file = summary_dir / "ingest_sdaia_summary.json"
    summary_file.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("SDAIA ingestion summary written to %s", summary_file)


# ---------------------------------------------------------------------------
# NCA (full crawl, target 500+ PDFs)
# ---------------------------------------------------------------------------

NCA_CORE_PDFS = [
    "https://cdn.nca.gov.sa/api/files/public/upload/86e09090-44e4-481f-bc28-355673607654_ECC--2024-EN.pdf",
    "https://cdn.nca.gov.sa/api/files/public/upload/6d5408a3-d8e6-4e96-963b-2c7198e5b7c2_CCC-2-2024-EN-.pdf",
    "https://cdn.nca.gov.sa/api/public/cms/files/f15af01c-dc59-4281-95e2-03a770655937_Critical-Systems-Cybersecurity-Controls.pdf",
]

NCA_CRAWL_CONFIG = {
    "base_urls": [
        "https://nca.gov.sa/en/",
        "https://nca.gov.sa/ar/",
        "https://csrc.nca.gov.sa/",
    ],
    "allowed_domains": ["nca.gov.sa", "csrc.nca.gov.sa", "cdn.nca.gov.sa"],
    "max_pages": 1500,
    "max_depth": 5,
    "target_pdfs": 500,
}


def discover_nca_pdfs(resume: bool = False, workers: int = 1) -> List[str]:
    logger.info("Crawling NCA for PDFs (target 500+)...")
    crawled = crawl_site.crawl_for_pdfs(
        base_urls=NCA_CRAWL_CONFIG["base_urls"],
        allowed_domains=NCA_CRAWL_CONFIG["allowed_domains"],
        max_pages=NCA_CRAWL_CONFIG["max_pages"],
        max_depth=NCA_CRAWL_CONFIG["max_depth"],
        target_pdfs=NCA_CRAWL_CONFIG["target_pdfs"],
        output_path=DISCOVERY_OUTPUT_DIR / "ingest_nca_discovered.json",
        resume=resume,
        workers=workers,
        request_delay=REQUEST_DELAY,
    )
    pdfs: Set[str] = set(NCA_CORE_PDFS) | set(crawled)
    return sorted(pdfs)


def ingest_nca(
    discover_only: bool = False,
    resume: bool = False,
    workers: int = 1,
) -> None:
    started_at = datetime.utcnow().isoformat() + "Z"
    stats: Dict[str, Any] = {
        "source": "nca",
        "started_at": started_at,
        "pdfs_discovered": 0,
        "pdfs_downloaded": 0,
        "pdfs_failed": 0,
        "pipeline_result": {},
    }

    pdf_urls = discover_nca_pdfs(resume=resume, workers=workers)
    stats["pdfs_discovered"] = len(pdf_urls)
    logger.info("Total NCA PDFs discovered: %s", len(pdf_urls))

    if discover_only:
        logger.info("Discovery only - skipping download and processing")
        return

    pdf_dir = BACKEND_DIR / "pdfs" / "nca"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    for url in pdf_urls:
        dest = pdf_dir / safe_filename_from_url(url)
        ok = download_with_requests(url, dest)
        if ok:
            stats["pdfs_downloaded"] += 1
        else:
            stats["pdfs_failed"] += 1

    logger.info("[PIPELINE] Processing NCA PDFs through RAG pipeline...")
    pipeline_result = process_pdfs_batch.process_directory(
        pdf_dir,
        output_dir=OUTPUT_DIR,
        chunks_dir=CHUNKS_OUTPUT_DIR,
        batch_size=CHUNK_BATCH_SIZE,
        skip_existing=False,
    )
    stats["pipeline_result"] = pipeline_result or {}
    stats["finished_at"] = datetime.utcnow().isoformat() + "Z"

    summary_dir = BACKEND_DIR / "output"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_file = summary_dir / "ingest_nca_summary.json"
    summary_file.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("NCA ingestion summary written to %s", summary_file)


# ---------------------------------------------------------------------------
# SAMA (reuse existing sama_crawler)
# ---------------------------------------------------------------------------

def ingest_sama() -> None:
    """
    Delegate to sama_crawler's existing logic.

    This will:
      - Discover SAMA rulebook pages
      - Collect PDF links
      - Download PDFs to backend/pdfs/sama
      - Process PDFs via process_pdf_to_supabase
    """
    logger.info("Ingesting SAMA using sama_crawler (all default sections)...")
    # Same default sections as sama_crawler --all-sections
    sections = [
        ("https://rulebook.sama.gov.sa/en/banking-sector-0", "Banking Sector"),
        ("https://rulebook.sama.gov.sa/en/finance-sector-0", "Finance Sector"),
        ("https://rulebook.sama.gov.sa/en/payment-systems-and-payment-services-providers", "Payment Systems"),
        ("https://rulebook.sama.gov.sa/en/money-exchange-sector-0", "Money Exchange Sector"),
        ("https://rulebook.sama.gov.sa/en/credit-bureaus", "Credit Bureaus"),
        ("https://rulebook.sama.gov.sa/en/regulatory-sandbox", "Regulatory Sandbox"),
    ]

    results = []
    for url, name in sections:
        r = sama_crawler.crawl_and_process_section(
            start_url=url,
            section_name=name,
            output_dir=OUTPUT_DIR,
            chunks_dir=CHUNKS_OUTPUT_DIR,
            batch_size=CHUNK_BATCH_SIZE,
        )
        results.append(r)

    ok = sum(1 for r in results if r.get("success"))
    logger.info("SAMA ingestion completed. Sections OK: %s/%s", ok, len(sections))


# ---------------------------------------------------------------------------
# Aramco (full crawl, target 500+ PDFs)
# ---------------------------------------------------------------------------

ARAMCO_CRAWL_CONFIG = {
    "base_urls": [
        "https://www.aramco.com/en/",
        "https://www.aramco.com/ar/",
    ],
    "allowed_domains": ["aramco.com", "www.aramco.com"],
    "max_pages": 2000,
    "max_depth": 6,
    "target_pdfs": 500,
}


def discover_aramco_pdfs(resume: bool = False, workers: int = 1) -> List[str]:
    logger.info("Crawling Aramco for PDFs (target 500+)...")
    return crawl_site.crawl_for_pdfs(
        base_urls=ARAMCO_CRAWL_CONFIG["base_urls"],
        allowed_domains=ARAMCO_CRAWL_CONFIG["allowed_domains"],
        max_pages=ARAMCO_CRAWL_CONFIG["max_pages"],
        max_depth=ARAMCO_CRAWL_CONFIG["max_depth"],
        target_pdfs=ARAMCO_CRAWL_CONFIG["target_pdfs"],
        output_path=DISCOVERY_OUTPUT_DIR / "ingest_aramco_discovered.json",
        resume=resume,
        workers=workers,
        request_delay=REQUEST_DELAY,
    )


def ingest_aramco(
    discover_only: bool = False,
    resume: bool = False,
    workers: int = 1,
) -> None:
    started_at = datetime.utcnow().isoformat() + "Z"
    stats: Dict[str, Any] = {
        "source": "aramco",
        "started_at": started_at,
        "pdfs_discovered": 0,
        "pdfs_downloaded": 0,
        "pdfs_failed": 0,
        "pipeline_result": {},
    }

    pdf_urls = discover_aramco_pdfs(resume=resume, workers=workers)
    stats["pdfs_discovered"] = len(pdf_urls)
    logger.info("Total Aramco PDFs discovered: %s", len(pdf_urls))

    if discover_only:
        logger.info("Discovery only - skipping download and processing")
        return

    pdf_dir = BACKEND_DIR / "pdfs" / "aramco"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    for url in pdf_urls:
        dest = pdf_dir / safe_filename_from_url(url)
        # Use curl because requests has TLS renegotiation issues here
        ok = download_with_curl(url, dest)
        if ok:
            stats["pdfs_downloaded"] += 1
        else:
            stats["pdfs_failed"] += 1

    logger.info("[PIPELINE] Processing Aramco PDFs through RAG pipeline...")
    pipeline_result = process_pdfs_batch.process_directory(
        pdf_dir,
        output_dir=OUTPUT_DIR,
        chunks_dir=CHUNKS_OUTPUT_DIR,
        batch_size=CHUNK_BATCH_SIZE,
        skip_existing=False,
    )
    stats["pipeline_result"] = pipeline_result or {}
    stats["finished_at"] = datetime.utcnow().isoformat() + "Z"

    summary_dir = BACKEND_DIR / "output"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_file = summary_dir / "ingest_aramco_summary.json"
    summary_file.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("Aramco ingestion summary written to %s", summary_file)


# ---------------------------------------------------------------------------
# ISO (full crawl, target 500+ PDFs)
# ---------------------------------------------------------------------------

ISO_CRAWL_CONFIG = {
    "base_urls": [
        "https://www.iso.org/",
    ],
    "allowed_domains": ["iso.org", "www.iso.org"],
    "max_pages": 2000,
    "max_depth": 5,
    "target_pdfs": 500,
}


def discover_iso_pdfs(resume: bool = False, workers: int = 1) -> List[str]:
    logger.info("Crawling ISO for PDFs (target 500+)...")
    return crawl_site.crawl_for_pdfs(
        base_urls=ISO_CRAWL_CONFIG["base_urls"],
        allowed_domains=ISO_CRAWL_CONFIG["allowed_domains"],
        max_pages=ISO_CRAWL_CONFIG["max_pages"],
        max_depth=ISO_CRAWL_CONFIG["max_depth"],
        target_pdfs=ISO_CRAWL_CONFIG["target_pdfs"],
        output_path=DISCOVERY_OUTPUT_DIR / "ingest_iso_discovered.json",
        resume=resume,
        workers=workers,
        request_delay=REQUEST_DELAY,
    )


def ingest_iso(
    discover_only: bool = False,
    resume: bool = False,
    workers: int = 1,
) -> None:
    started_at = datetime.utcnow().isoformat() + "Z"
    stats: Dict[str, Any] = {
        "source": "iso",
        "started_at": started_at,
        "pdfs_discovered": 0,
        "pdfs_downloaded": 0,
        "pdfs_failed": 0,
        "pipeline_result": {},
    }

    pdf_urls = discover_iso_pdfs(resume=resume, workers=workers)
    stats["pdfs_discovered"] = len(pdf_urls)
    logger.info("Total ISO PDFs discovered: %s", len(pdf_urls))

    if discover_only:
        logger.info("Discovery only - skipping download and processing")
        return

    pdf_dir = BACKEND_DIR / "pdfs" / "iso"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    for url in pdf_urls:
        dest = pdf_dir / safe_filename_from_url(url)
        ok = download_with_requests(url, dest)
        if ok:
            stats["pdfs_downloaded"] += 1
        else:
            stats["pdfs_failed"] += 1

    logger.info("[PIPELINE] Processing ISO PDFs through RAG pipeline...")
    pipeline_result = process_pdfs_batch.process_directory(
        pdf_dir,
        output_dir=OUTPUT_DIR,
        chunks_dir=CHUNKS_OUTPUT_DIR,
        batch_size=CHUNK_BATCH_SIZE,
        skip_existing=False,
    )
    stats["pipeline_result"] = pipeline_result or {}
    stats["finished_at"] = datetime.utcnow().isoformat() + "Z"

    summary_dir = BACKEND_DIR / "output"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_file = summary_dir / "ingest_iso_summary.json"
    summary_file.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("ISO ingestion summary written to %s", summary_file)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest regulations PDFs into Supabase")
    parser.add_argument(
        "--source",
        type=str,
        default="all",
        choices=["all", "sdaia", "nca", "sama", "aramco", "iso"],
        help="Which source to ingest (default: all)",
    )
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Only run discovery (crawl for PDF URLs), skip download and processing",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load discovered URLs from saved JSON instead of re-crawling",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Parallel crawl workers (default: 1)",
    )
    args = parser.parse_args()

    # NOTE: SAMA is intentionally excluded from the 'all' aggregate run
    # because it has already been fully ingested separately.

    if args.source in ("all", "sdaia"):
        ingest_sdaia(
            discover_only=args.discover_only,
            resume=args.resume,
            workers=args.workers,
        )

    if args.source in ("all", "nca"):
        ingest_nca(
            discover_only=args.discover_only,
            resume=args.resume,
            workers=args.workers,
        )

    if args.source == "sama":
        ingest_sama()

    if args.source in ("all", "aramco"):
        ingest_aramco(
            discover_only=args.discover_only,
            resume=args.resume,
            workers=args.workers,
        )

    if args.source in ("all", "iso"):
        ingest_iso(
            discover_only=args.discover_only,
            resume=args.resume,
            workers=args.workers,
        )


if __name__ == "__main__":
    main()


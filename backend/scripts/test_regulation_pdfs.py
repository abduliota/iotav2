"""
Quick smoke test for public regulations PDFs and index pages.

This script:
  - Checks a curated list of direct PDF URLs from SDAIA, NCA, SAMA, Aramco, and ISO.
  - Checks key HTML index/listing pages and verifies that they expose at least one .pdf link.

It does NOT download or persist files; it only makes HTTP GET requests and reports:
  - HTTP status code
  - Content-Type
  - Approximate content length
  - For HTML pages, discovered .pdf links

Usage (from backend/):

    python scripts/test_regulation_pdfs.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from typing import Any, Dict, List
from urllib.parse import urljoin

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("Error: Missing dependencies. Install with: pip install requests beautifulsoup4")
    sys.exit(1)


TIMEOUT = 60  # total timeout budget per request (seconds)
MAX_RETRIES = 3


def check_with_curl_head(url: str, timeout: int = 60) -> bool:
    """
    Use curl to perform a lightweight HEAD request.

    This is used for Aramco URLs where Python's TLS stack intermittently times out,
    but curl (Schannel on Windows) works reliably.
    """
    try:
        result = subprocess.run(
            ["curl", "-I", "--max-time", str(timeout), url],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


PDF_URLS: Dict[str, List[str]] = {
    "SDAIA": [
        # Personal Data Protection Law (English)
        "https://sdaia.gov.sa/en/SDAIA/about/Documents/Personal%20Data%20English%20V2-23April2023-%20Reviewed-.pdf",
        # Implementing Regulation of the PDPL
        "https://sdaia.gov.sa/en/SDAIA/about/Documents/ImplementingRegulation.pdf",
        # Common Rules / BCR for personal data transfer
        "https://sdaia.gov.sa/Documents/CommonRulesBCRForPersonalDataTransferEN.pdf",
        # Executive regulations and rules governing the national register (if present)
        "https://sdaia.gov.sa/en/SDAIA/about/Documents/ExecutiveRegulations.pdf",
        "https://sdaia.gov.sa/ar/SDAIA/eParticipation/Files/Rules-Governing-en.pdf",
        # Generative AI guidelines for government
        "https://sdaia.gov.sa/en/SDAIA/about/Files/GenAIGuidelinesForGovernmentENCompressed.pdf",
    ],
    "NCA": [
        # Essential Cybersecurity Controls (ECC-2:2024)
        "https://cdn.nca.gov.sa/api/files/public/upload/86e09090-44e4-481f-bc28-355673607654_ECC--2024-EN.pdf",
        # Cloud Cybersecurity Controls (CCC-2:2024)
        "https://cdn.nca.gov.sa/api/files/public/upload/6d5408a3-d8e6-4e96-963b-2c7198e5b7c2_CCC-2-2024-EN-.pdf",
        # Critical Systems Cybersecurity Controls
        "https://cdn.nca.gov.sa/api/public/cms/files/f15af01c-dc59-4281-95e2-03a770655937_Critical-Systems-Cybersecurity-Controls.pdf",
    ],
    "SAMA": [
        # Oversight framework for payments and financial settlement systems
        "https://www.sama.gov.sa/en-US/RulesInstructions/Documents/Oversight_Framework_for_Payments_and_Financial_Settlement_Systems-EN.pdf",
        # Saudi Central Bank Law (English)
        "https://www.sama.gov.sa/en-US/Documents/SCB-EN.pdf",
    ],
    "ARAMCO": [
        # CCC program core documents
        "https://www.aramco.com/-/media/downloads/working-with-us/ccc/ccc-third-party-manual.pdf",
        "https://www.aramco.com/-/media/downloads/working-with-us/ccc/cybersecurity-compliance-certificate-ccc-audit-firms.pdf",
        "https://www.aramco.com/-/media/downloads/working-with-us/ccc/cybersecurity-controls-requirements-guideline.pdf",
        "https://www.aramco.com/-/media/downloads/working-with-us/ccc/sacs-002-third-party-cybersecurity-standard.pdf",
        # Supplier / quality governance documents
        "https://www.aramco.com/-/media/downloads/working-with-us/ertqa.pdf",
        "https://www.aramco.com/-/media/downloads/working-with-us/become-a-supplier/saudi-aramco-supplier-code-of-conduct_en.pdf",
    ],
    "ISO_PUBLIC": [
        # International Classification for Standards (ICS) – public reference document
        "https://www.iso.org/files/live/sites/isoorg/files/archive/pdf/en/international_classification_for_standards.pdf",
    ],
}


PAGE_URLS: Dict[str, List[str]] = {
    "SDAIA_INDEX": [
        "https://sdaia.gov.sa/en/SDAIA/about/Pages/RegulationsAndPolicies.aspx",
        "https://sdaia.gov.sa/en/Research/Pages/DataProtection.aspx",
    ],
    "NCA_INDEX": [
        "https://nca.gov.sa/en/regulatory-documents/",
        "https://nca.gov.sa/en/regulatory-documents/?documentType=frameworks-and-standard-list",
        "https://nca.gov.sa/en/regulatory-documents/guidelines-list/cybersecurity-toolkits/",
    ],
    "SAMA_INDEX": [
        "https://rulebook.sama.gov.sa/en/regulations-and-instructions",
        "https://rulebook.sama.gov.sa/en/rules-and-instructions-0",
    ],
    "ARAMCO_INDEX": [
        "https://www.aramco.com/en/what-we-do/suppliers/supplier-resources/cybersecurity-compliance-certificate-program",
    ],
    "ISO_INDEX": [
        "https://www.iso.org/standards.html",
        "https://www.iso.org/standards-catalogue/browse-by-ics.html",
    ],
}


def _headers() -> Dict[str, str]:
    # Use browser-like headers to reduce chance of WAF/CDN throttling.
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


def fetch(url: str, stream: bool = False) -> Dict[str, Any]:
    """Fetch a URL with basic retry logic and return a small metadata dict.

    When stream=True, only a limited amount of data is read from the body
    (sufficient to validate reachability and type) to avoid long timeouts
    on large PDFs or slow connections.
    """
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Use separate connect/read timeouts within the overall TIMEOUT budget.
            resp = requests.get(
                url,
                headers=_headers(),
                timeout=(10, TIMEOUT),
                allow_redirects=True,
                stream=stream,
            )
            content_type = resp.headers.get("Content-Type", "")

            if stream:
                # Read only up to max_bytes to confirm the resource is alive.
                bytes_read = 0
                max_bytes = 64 * 1024  # 64 KB
                for chunk in resp.iter_content(chunk_size=8192):
                    if not chunk:
                        break
                    bytes_read += len(chunk)
                    if bytes_read >= max_bytes:
                        break
                length = bytes_read
            else:
                # Prefer Content-Length header if present; fall back to response body length.
                if "Content-Length" in resp.headers:
                    try:
                        length = int(resp.headers["Content-Length"])
                    except ValueError:
                        length = len(resp.content or b"")
                else:
                    length = len(resp.content or b"")
            return {
                "ok": resp.ok,
                "status": resp.status_code,
                "content_type": content_type,
                "length": length,
                "error": None,
                "response": resp,
            }
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                time.sleep(2)
                continue
            return {
                "ok": False,
                "status": None,
                "content_type": "",
                "length": 0,
                "error": str(last_exc),
                "response": None,
            }


def test_pdf_urls() -> None:
    """Check that direct PDF URLs are reachable and actually look like PDFs."""
    print("=== TESTING DIRECT PDF URLS ===")
    for group, urls in PDF_URLS.items():
        print(f"\n--- {group} ---")
        for url in urls:
            if group == "ARAMCO":
                ok = check_with_curl_head(url)
                if ok:
                    print(f"[OK] {url}\n       via curl HEAD")
                else:
                    print(f"[FAIL] {url}\n       error=curl HEAD returned non-zero exit code")
                continue

            # For large PDFs, stream to avoid downloading the entire body during this smoke test.
            use_stream = group in {"SDAIA", "NCA", "SAMA"}
            result = fetch(url, stream=use_stream)
            if not result["ok"]:
                print(f"[FAIL] {url}\n       error={result['error']}")
                continue

            ct = result["content_type"].lower()
            is_pdf = "pdf" in ct
            status = result["status"]
            length = result["length"]
            flag = "OK" if (status == 200 and is_pdf and length > 0) else "WARN"
            print(
                f"[{flag}] {url}\n"
                f"       status={status}, type='{ct}', bytes={length}"
            )


def extract_pdfs_from_html(url: str) -> List[str]:
    """Fetch an HTML page and return a list of discovered .pdf links (absolute URLs)."""
    result = fetch(url)
    if not result["ok"]:
        print(f"[FAIL] {url}\n       error={result['error']}")
        return []

    resp = result["response"]
    assert resp is not None
    ct = result["content_type"].lower()
    if "html" not in ct:
        print(f"[WARN] {url}\n       Content-Type is not HTML ('{ct}'), skipping parse")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    pdf_links: set[str] = set()

    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        if href.lower().endswith(".pdf"):
            full = urljoin(resp.url, href)
            pdf_links.add(full)

    return sorted(pdf_links)


def test_page_urls() -> None:
    """Check that key index/listing pages expose at least one .pdf link."""
    print("\n=== TESTING HTML INDEX PAGES FOR PDF LINKS ===")
    for group, urls in PAGE_URLS.items():
        print(f"\n--- {group} ---")
        # For Aramco, just verify reachability via curl HEAD to avoid TLS issues when parsing.
        if group == "ARAMCO_INDEX":
            for url in urls:
                print(f"\nPage: {url}")
                ok = check_with_curl_head(url)
                if ok:
                    print("  [INFO] Aramco index reachable via curl HEAD.")
                else:
                    print("  [FAIL] Aramco index not reachable via curl HEAD.")
            continue
        for url in urls:
            print(f"\nPage: {url}")
            pdfs = extract_pdfs_from_html(url)
            if not pdfs:
                print("  [INFO] No .pdf links found on this page.")
            else:
                print(f"  [INFO] Found {len(pdfs)} PDF link(s):")
                for p in pdfs:
                    print(f"    - {p}")


def main() -> None:
    """CLI entry point."""
    test_pdf_urls()
    test_page_urls()


if __name__ == "__main__":
    main()


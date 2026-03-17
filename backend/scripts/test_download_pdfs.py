"""
Test downloading key regulations PDFs into a dedicated test folder.

This script uses the same URL sets as the smoke test to:
  - Download each PDF into backend/pdfs/test_downloads/<group>/
  - Use requests for SDAIA/NCA/SAMA/ISO
  - Use curl for Aramco (due to TLS renegotiation issues with requests)

Usage (from backend/):

    python scripts/test_download_pdfs.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

try:
    import requests
except ImportError:
    print("Error: Missing dependency 'requests'. Install with: pip install requests")
    sys.exit(1)


PDF_URLS: Dict[str, List[str]] = {
    "SDAIA": [
        "https://sdaia.gov.sa/en/SDAIA/about/Documents/Personal%20Data%20English%20V2-23April2023-%20Reviewed-.pdf",
        "https://sdaia.gov.sa/en/SDAIA/about/Documents/ImplementingRegulation.pdf",
        "https://sdaia.gov.sa/Documents/CommonRulesBCRForPersonalDataTransferEN.pdf",
        "https://sdaia.gov.sa/en/SDAIA/about/Documents/ExecutiveRegulations.pdf",
        "https://sdaia.gov.sa/ar/SDAIA/eParticipation/Files/Rules-Governing-en.pdf",
        "https://sdaia.gov.sa/en/SDAIA/about/Files/GenAIGuidelinesForGovernmentENCompressed.pdf",
    ],
    "NCA": [
        "https://cdn.nca.gov.sa/api/files/public/upload/86e09090-44e4-481f-bc28-355673607654_ECC--2024-EN.pdf",
        "https://cdn.nca.gov.sa/api/files/public/upload/6d5408a3-d8e6-4e96-963b-2c7198e5b7c2_CCC-2-2024-EN-.pdf",
        "https://cdn.nca.gov.sa/api/public/cms/files/f15af01c-dc59-4281-95e2-03a770655937_Critical-Systems-Cybersecurity-Controls.pdf",
    ],
    "SAMA": [
        "https://www.sama.gov.sa/en-US/RulesInstructions/Documents/Oversight_Framework_for_Payments_and_Financial_Settlement_Systems-EN.pdf",
        "https://www.sama.gov.sa/en-US/Documents/SCB-EN.pdf",
    ],
    "ARAMCO": [
        "https://www.aramco.com/-/media/downloads/working-with-us/ccc/ccc-third-party-manual.pdf",
        "https://www.aramco.com/-/media/downloads/working-with-us/ccc/cybersecurity-compliance-certificate-ccc-audit-firms.pdf",
        "https://www.aramco.com/-/media/downloads/working-with-us/ccc/cybersecurity-controls-requirements-guideline.pdf",
        "https://www.aramco.com/-/media/downloads/working-with-us/ccc/sacs-002-third-party-cybersecurity-standard.pdf",
        "https://www.aramco.com/-/media/downloads/working-with-us/ertqa.pdf",
        "https://www.aramco.com/-/media/downloads/working-with-us/become-a-supplier/saudi-aramco-supplier-code-of-conduct_en.pdf",
    ],
    "ISO_PUBLIC": [
        "https://www.iso.org/files/live/sites/isoorg/files/archive/pdf/en/international_classification_for_standards.pdf",
    ],
}


BACKEND_DIR = Path(__file__).resolve().parent.parent
TEST_DOWNLOAD_DIR = BACKEND_DIR / "pdfs" / "test_downloads"
TEST_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "application/pdf,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }


def safe_filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name or "file.pdf"
    # Basic sanitization for Windows file names.
    for ch in '<>:"/\\|?*':
        name = name.replace(ch, "_")
    return name or "file.pdf"


def download_with_requests(url: str, dest: Path, timeout: int = 60) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [SKIP] {dest.name} already exists ({dest.stat().st_size} bytes)")
        return True

    try:
        with requests.get(
            url,
            headers=_headers(),
            timeout=(10, timeout),
            stream=True,
        ) as r:
            if r.status_code != 200:
                print(f"  [FAIL] {url} -- status={r.status_code}")
                return False
            ct = r.headers.get("Content-Type", "").lower()
            if "pdf" not in ct:
                print(f"  [WARN] {url} -- unexpected Content-Type '{ct}'")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
        print(f"  [OK] downloaded {dest.name} ({dest.stat().st_size} bytes)")
        return True
    except requests.RequestException as exc:
        print(f"  [FAIL] {url} -- error={exc}")
        return False


def download_with_curl(url: str, dest: Path, timeout: int = 120) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [SKIP] {dest.name} already exists ({dest.stat().st_size} bytes)")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["curl", "-L", "--max-time", str(timeout), "-o", str(dest), url]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
        print(f"  [OK] downloaded {dest.name} via curl ({dest.stat().st_size} bytes)")
        return True

    print(
        f"  [FAIL] {url} -- curl exit={result.returncode}, "
        f"stderr={result.stderr.strip()}"
    )
    return False


def main() -> None:
    print(f"Test download directory: {TEST_DOWNLOAD_DIR}")

    if shutil.which("curl") is None:
        print("Warning: 'curl' not found in PATH. Aramco downloads will fail.")

    for group, urls in PDF_URLS.items():
        group_dir = TEST_DOWNLOAD_DIR / group.lower()
        print(f"\n=== {group} ===")
        for url in urls:
            filename = safe_filename_from_url(url)
            dest = group_dir / filename
            if group == "ARAMCO":
                download_with_curl(url, dest)
            else:
                download_with_requests(url, dest)


if __name__ == "__main__":
    main()


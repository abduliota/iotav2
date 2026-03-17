from __future__ import annotations

"""
Minimal configuration for the IOTAV3 RAG backend.

This module is intentionally much smaller and more focused than the
legacy `backend/config.py`. It only exposes the settings that the
IOTAV3‑native RAG pipeline and context engine need:

- Brand / persona information
- Domain and guardrail configuration
- Retrieval and source settings
- Cache backend configuration
"""

import os
from pathlib import Path
from typing import List


def _env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "y")


def _env_int(key: str, default: int) -> int:
    v = os.getenv(key)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_str(key: str, default: str) -> str:
    return os.getenv(key, default)


def _env_float(key: str, default: float) -> float:
    v = os.getenv(key)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


BACKEND_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Brand / persona
# ---------------------------------------------------------------------------

APP_BRAND_NAME: str = _env_str(
    "IOTAV3_APP_BRAND_NAME",
    "Saudi GRC Compliance Assistant",
)

# ---------------------------------------------------------------------------
# Domain and guardrails
# ---------------------------------------------------------------------------

# Pipe‑ or comma‑separated list of keywords that indicate the query is
# in‑scope for this assistant (SAMA, NORA, Aramco CCC, NCA ECC, PDPL,
# ISO 27k, etc.).
_IN_SCOPE_RAW = _env_str(
    "IOTAV3_IN_SCOPE_KEYWORDS",
    (
        "sama|saudi central bank|nora|aramco|ccc|"
        "national cybersecurity authority|nca|ecc|"
        "pdpl|personal data protection law|"
        "iso 27001|iso/iec 27001|information security management system"
    ),
)
IN_SCOPE_KEYWORDS: List[str] = [
    s.strip().lower()
    for part in _IN_SCOPE_RAW.split(",")
    for s in part.split("|")
    if s.strip()
]

# Short anchor sentences used by any semantic domain gate to recognise
# relevant questions. These are deliberately high‑level; the retrieval
# layer remains responsible for fine‑grained similarity.
DOMAIN_ANCHOR_PHRASES: List[str] = [
    "SAMA and NORA banking regulations and supervision in Saudi Arabia",
    "Aramco Cybersecurity Compliance Certificate program requirements",
    "National Cybersecurity Authority Essential Cybersecurity Controls in Saudi Arabia",
    "Saudi Personal Data Protection Law PDPL obligations and rights",
    "ISO 27001 information security management system controls and clauses",
]

# Phrases that clearly indicate an off‑topic question where we should
# refuse to answer regardless of available context.
_OFF_TOPIC_RAW = _env_str(
    "IOTAV3_OFF_TOPIC_PATTERNS",
    (
        "us president|who is the president|weather|sports|football|"
        "movie|recipe|stock market|crypto price|bitcoin|ethereum"
    ),
)
OFF_TOPIC_PATTERNS: List[str] = [
    s.strip().lower()
    for part in _OFF_TOPIC_RAW.split(",")
    for s in part.split("|")
    if s.strip()
]

OUT_OF_SCOPE_MESSAGE: str = _env_str(
    "IOTAV3_OUT_OF_SCOPE_MESSAGE",
    (
        "I can only answer questions about Saudi GRC frameworks such as "
        "SAMA, NORA, Aramco CCC, NCA ECC, PDPL, and relevant ISO standards."
    ),
)

NOT_FOUND_MESSAGE: str = _env_str(
    "IOTAV3_NOT_FOUND_MESSAGE",
    (
        "The information was not found in the configured Saudi GRC documents "
        "(SAMA, NORA, Aramco CCC, NCA ECC, PDPL, ISO 27k)."
    ),
)

# ---------------------------------------------------------------------------
# Retrieval / sources
# ---------------------------------------------------------------------------

TOP_K: int = _env_int("IOTAV3_TOP_K", 5)

# Number of chunks to retrieve for definition-style queries (e.g. "what is SAMA?").
TOP_K_DEFINITION: int = _env_int("IOTAV3_TOP_K_DEFINITION", 15)

MAX_SOURCES: int = _env_int("IOTAV3_MAX_SOURCES", 5)

# Maximum characters to keep per snippet returned from the Supabase
# `match_chunks` RPC. This mirrors (in a simplified form) the legacy
# `SNIPPET_CHAR_LIMIT` constant used by `backend/simple_rag.py`.
SNIPPET_CHAR_LIMIT: int = _env_int("IOTAV3_SNIPPET_CHAR_LIMIT", 1800)

# Acronym/term expansions used only when building the query string for
# embedding (e.g. "what is SAMA?" -> append "Saudi Arabian Monetary Authority"
# so retrieval ranks definition chunks higher). Keys are lowercased terms.
QUERY_EXPANSION_FOR_EMBEDDING: dict[str, str] = {
    "sama": "Saudi Arabian Monetary Authority",
    "nora": "National Regulatory Authority for Islamic banks Saudi Arabia",
}

# Minimum similarity for retrieval to be considered a strong in-domain
# signal even if no explicit scope keyword is present.
RETRIEVAL_IN_SCOPE_MIN_SIM: float = _env_float(
    "IOTAV3_RETRIEVAL_IN_SCOPE_MIN_SIM",
    0.30,
)

# Minimum similarity for a passage to be treated as a usable piece of
# context when deciding between NOT_FOUND and OUT_OF_SCOPE.
RETRIEVAL_MIN_CONTEXT_SIM: float = _env_float(
    "IOTAV3_RETRIEVAL_MIN_CONTEXT_SIM",
    0.25,
)

# ---------------------------------------------------------------------------
# Cache configuration
# ---------------------------------------------------------------------------

CACHE_ENABLED: bool = _env_bool("CACHE_ENABLED", True)
CACHE_BACKEND: str = _env_str("CACHE_BACKEND", "memory")
CACHE_TTL_SECONDS: int = _env_int("CACHE_TTL_SECONDS", 3600)
REDIS_URL: str = _env_str("REDIS_URL", "")

# Optional experimental guard that can later be used to enforce that key
# entities from the question appear in the chosen passages before
# answering. Currently unused but reserved for future hardening.
ENTITY_GUARD_ENABLED: bool = _env_bool("IOTAV3_ENTITY_GUARD_ENABLED", False)


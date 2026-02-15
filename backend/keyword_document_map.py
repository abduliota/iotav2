"""Keyword → document/cluster mapping for retrieval boost (e.g. SAMA EN documents)."""
from __future__ import annotations

import re
from typing import Any

# Keywords/phrases (lowercase) -> list of preferred document name substrings (match if in document_name)
# Plan: digital-bank → SAMA EN 1897; related parties → connected lending; remuneration → SAMA EN 2926;
# Shariah governance → SAMA EN 2274; cyber → SAMA EN 5888; depositor protection → SAMA EN 3623;
# fit-and-proper → Licensing; foreign branches → SAMA EN 1713; penalties → BCL + Circular;
# capital requirement → Basel; AML reporting → AML/CTF.
KEYWORD_TO_DOCUMENTS: dict[str, list[str]] = {
    "digital bank": ["1897", "digital bank"],
    "digital banking": ["1897", "digital"],
    "deposit insurance": ["deposit insurance", "3623", "depositor"],
    "depositor protection": ["3623", "depositor"],
    "related parties": ["related part", "connected lending", "lending"],
    "remuneration": ["2926", "remuneration"],
    "shariah governance": ["2274", "shariah", "governance"],
    "cyber": ["5888", "cyber", "cybersecurity"],
    "fit and proper": ["licensing", "fit", "proper"],
    "fit-and-proper": ["licensing", "fit", "proper"],
    "foreign branches": ["1713", "foreign branch"],
    "penalties": ["penalty", "bcl", "circular"],
    "capital requirement": ["basel", "capital"],
    "capital adequacy": ["basel", "capital"],
    "aml": ["aml", "ctf", "anti-money", "combating financing"],
    "aml reporting": ["aml", "ctf", "guide"],
    "licensing": ["licensing", "license"],
    "sama": ["sama", "saudi arabian monetary"],
    "nora": ["nora", "national"],
}


def _normalize_query_for_keywords(q: str) -> str:
    """Lowercase and collapse whitespace for keyword matching."""
    return " ".join(re.split(r"\s+", (q or "").lower().strip()))


def get_documents_for_query(query: str) -> list[str]:
    """
    Return list of preferred document name substrings for the query.
    Used to boost chunks whose document_name contains any of these strings.
    """
    if not query or not query.strip():
        return []
    normalized = _normalize_query_for_keywords(query)
    preferred: list[str] = []
    for keywords, doc_hints in KEYWORD_TO_DOCUMENTS.items():
        if keywords in normalized:
            preferred.extend(doc_hints)
    # Deduplicate while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for s in preferred:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def chunk_matches_preferred_documents(chunk: dict[str, Any], preferred: list[str]) -> bool:
    """True if chunk's document_name contains any of the preferred substrings (case-insensitive)."""
    if not preferred:
        return False
    doc_name = (chunk.get("document_name") or "").lower()
    return any(hint.lower() in doc_name for hint in preferred)

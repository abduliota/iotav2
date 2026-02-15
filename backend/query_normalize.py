"""Query normalization before embedding: synonyms, legal terms, acronyms.
Run before embedding in simple_rag to improve retrieval for short and domain queries.
"""
from __future__ import annotations

import re
from typing import Any

# Domain synonym expansions: term -> list of additional terms to help retrieval
# Format: lowercase key -> space-separated expansion words (appended to query for retrieval)
DOMAIN_SYNONYMS: dict[str, str] = {
    "license": "authorization permit banking license application",
    "licensing": "authorization permit licensing requirements",
    "aml": "anti-money laundering",
    "kyc": "know your customer identification",
    "fit and proper": "qualifications suitability criteria",
    "remuneration": "compensation pay executive remuneration",
    "capital": "capital adequacy regulatory capital",
    "shariah": "shariah compliant islamic finance",
    "branch": "branch opening foreign branch",
    "outsourcing": "outsourcing third party",
    "ترخيص": "ترخيص تراخيص رخصة",
    "غسيل أموال": "مكافحة غسل الأموال",
    "شرعة": "الشرعة الإسلامية",
}

# Legal/regulatory term normalization: acronym or short form <-> full form (both directions help)
# We expand to full form so vector search matches document phrasing.
LEGAL_TERM_MAP: dict[str, str] = {
    "aml": "anti-money laundering",
    "anti-money laundering": "aml",
    "kyc": "know your customer",
    "know your customer": "kyc",
    "car": "capital adequacy ratio",
    "capital adequacy ratio": "car",
    "bcbs": "basel committee",
    "basel committee": "bcbs",
    "cbb": "central bank",
    "sama": "saudi arabian monetary authority",
    "nora": "national regulatory authority",
    "bcr": "banking control law",
    "banking control law": "bcr",
}

# Acronym resolver: acronym -> expanded form (for query expansion before embedding)
ACRONYM_EXPANSIONS: dict[str, str] = {
    "aml": "anti-money laundering",
    "kyc": "know your customer",
    "car": "capital adequacy ratio",
    "bcbs": "basel committee on banking supervision",
    "cbb": "central bank",
    "sama": "saudi arabian monetary authority",
    "bcr": "banking control regulation",
    "rr": "regulation",
    "cir": "circular",
    "cfr": "capital adequacy framework",
    "icaap": "internal capital adequacy assessment process",
    "rwa": "risk weighted assets",
    "cet1": "common equity tier 1",
    "tlac": "total loss absorbing capacity",
    "lcr": "liquidity coverage ratio",
    "nsfr": "net stable funding ratio",
}


def _load_overrides_from_config() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Load synonym, legal, acronym overrides from config if available; else use built-in defaults."""
    try:
        from config import (
            QUERY_NORMALIZE_ACRONYMS,
            QUERY_NORMALIZE_LEGAL_TERMS,
            QUERY_NORMALIZE_SYNONYMS,
        )
        syn = QUERY_NORMALIZE_SYNONYMS if QUERY_NORMALIZE_SYNONYMS else DOMAIN_SYNONYMS
        legal = QUERY_NORMALIZE_LEGAL_TERMS if QUERY_NORMALIZE_LEGAL_TERMS else LEGAL_TERM_MAP
        acr = QUERY_NORMALIZE_ACRONYMS if QUERY_NORMALIZE_ACRONYMS else ACRONYM_EXPANSIONS
        return syn, legal, acr
    except ImportError:
        return DOMAIN_SYNONYMS, LEGAL_TERM_MAP, ACRONYM_EXPANSIONS


def normalize_query(
    query: str,
    *,
    expand_synonyms: bool = True,
    expand_legal_terms: bool = True,
    expand_acronyms: bool = True,
    synonym_merge: str = " ",
) -> str:
    """
    Normalize and expand query for better retrieval.
    - expand_synonyms: append domain synonym phrases for recognized terms
    - expand_legal_terms: replace acronyms/short forms with full form (or add both)
    - expand_acronyms: expand known acronyms to full form
    Returns a single string with original query + expansions (suitable for embedding or keyword search).
    """
    if not query or not query.strip():
        return query
    q = query.strip()
    syn_dict, legal_dict, acronym_dict = _load_overrides_from_config()
    parts: list[str] = [q]
    seen: set[str] = set()

    def add_once(phrase: str) -> None:
        phrase = phrase.strip().lower()
        if phrase and phrase not in seen:
            seen.add(phrase)
            parts.append(phrase)

    q_lower = q.lower()
    tokens = set(re.findall(r"\b\w+\b", q_lower))

    if expand_acronyms and acronym_dict:
        for acr, expanded in acronym_dict.items():
            if acr in tokens or acr in q_lower:
                add_once(expanded)

    if expand_legal_terms and legal_dict:
        for short, full in legal_dict.items():
            if short in tokens or short in q_lower:
                add_once(full)

    if expand_synonyms and syn_dict:
        for term, expansion in syn_dict.items():
            if term in q_lower or term in tokens:
                for w in expansion.split():
                    add_once(w)

    merged = (q + synonym_merge + synonym_merge.join(parts[1:])).strip() if len(parts) > 1 else q
    # Collapse repeated spaces
    return re.sub(r"\s+", " ", merged).strip()


# Short regulatory queries (1-4 words): append expansion phrase for better retrieval
SHORT_QUERY_EXPANSIONS: dict[str, str] = {
    "license": "banking license application criteria requirements",
    "licensing": "banking licensing requirements application",
    "remuneration": "remuneration rules executive compensation",
    "aml": "anti-money laundering requirements",
    "capital": "capital adequacy requirements",
    "ترخيص": "ترخيص بنكي متطلبات",
    "مكافآت": "مكافآت تنفيذية",
}


def expand_short_query(query: str, max_words: int = 4) -> str:
    """
    If query has few words and matches a known regulatory term, append expansion phrase.
    Used before embedding to improve retrieval for short queries.
    """
    if not query or not query.strip():
        return query
    q = query.strip()
    words = len(q.split())
    if words > max_words:
        return q
    q_lower = q.lower()
    for term, expansion in SHORT_QUERY_EXPANSIONS.items():
        if term in q_lower or term in q:
            return (q + " " + expansion).strip()
    return q


def normalize_query_for_embedding(query: str, config: dict[str, Any] | None = None) -> str:
    """
    Single entry point: normalize and optionally expand short query before passing to embed_query.
    Uses config flags if provided (e.g. ENABLE_QUERY_NORMALIZE_SYNONYMS, ENABLE_QUERY_EXPANSION).
    """
    if not query or not query.strip():
        return query
    q = query.strip()
    expand_short = False
    if config is None:
        try:
            from config import (
                ENABLE_QUERY_EXPANSION,
                ENABLE_QUERY_NORMALIZE_ACRONYMS,
                ENABLE_QUERY_NORMALIZE_LEGAL,
                ENABLE_QUERY_NORMALIZE_SYNONYMS,
            )
            expand_short = ENABLE_QUERY_EXPANSION
            q = normalize_query(
                q,
                expand_synonyms=ENABLE_QUERY_NORMALIZE_SYNONYMS,
                expand_legal_terms=ENABLE_QUERY_NORMALIZE_LEGAL,
                expand_acronyms=ENABLE_QUERY_NORMALIZE_ACRONYMS,
            )
        except ImportError:
            q = normalize_query(q)
    else:
        expand_short = config.get("ENABLE_QUERY_EXPANSION", True)
        q = normalize_query(
            q,
            expand_synonyms=config.get("ENABLE_QUERY_NORMALIZE_SYNONYMS", True),
            expand_legal_terms=config.get("ENABLE_QUERY_NORMALIZE_LEGAL", True),
            expand_acronyms=config.get("ENABLE_QUERY_NORMALIZE_ACRONYMS", True),
        )
    if expand_short:
        q = expand_short_query(q)
    return q

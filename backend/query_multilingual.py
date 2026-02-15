"""Multilingual retrieval: Arabic detection, optional translation, dual retrieve + RRF merge."""
from __future__ import annotations

from typing import Any

# Arabic script range
_ARABIC_RE = __import__("re").compile(r"[\u0600-\u06FF]")


def is_arabic_query(text: str) -> bool:
    """True if text contains Arabic script (U+0600–U+06FF)."""
    if not text or not text.strip():
        return False
    return bool(_ARABIC_RE.search(text))


def translate_arabic_to_english(text: str) -> str | None:
    """
    Optional: translate Arabic query to English for dual retrieval.
    Returns translated string or None if translation disabled/fails.
    Uses OpenAI chat completion with a simple instruction when OPENAI_API_KEY is set.
    """
    if not text or not text.strip() or not is_arabic_query(text):
        return None
    try:
        from config import ENABLE_ARABIC_TRANSLATE_FOR_RETRIEVAL
        if not ENABLE_ARABIC_TRANSLATE_FOR_RETRIEVAL:
            return None
    except ImportError:
        return None
    try:
        from openai import OpenAI
        client = OpenAI()
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Translate the following Arabic text to English. Output only the translation, no explanation. Keep regulatory and banking terms accurate."},
                {"role": "user", "content": text.strip()},
            ],
            max_tokens=300,
        )
        if r.choices and r.choices[0].message and r.choices[0].message.content:
            return r.choices[0].message.content.strip() or None
    except Exception:
        pass
    return None


def _chunk_key(chunk: dict[str, Any]) -> tuple[str, int, int]:
    """Stable key for RRF deduplication."""
    doc = chunk.get("document_name") or ""
    start = int(chunk.get("page_start") or 0)
    end = int(chunk.get("page_end") or 0)
    return (doc, start, end)


def merge_chunks_rrf(
    ranked_lists: list[list[dict[str, Any]]],
    rrf_k: int = 60,
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    """
    Reciprocal Rank Fusion over multiple ranked lists of chunks.
    Each chunk dict must have document_name, page_start, page_end (used for deduplication).
    Returns merged list sorted by RRF score descending; at most top_n if set.
    """
    if not ranked_lists:
        return []
    scores: dict[tuple[str, int, int], float] = {}
    chunk_by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
    for rank_list in ranked_lists:
        for rank, chunk in enumerate(rank_list, start=1):
            key = _chunk_key(chunk)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            if key not in chunk_by_key:
                chunk_by_key[key] = chunk
    sorted_keys = sorted(scores.keys(), key=lambda k: -scores[k])
    if top_n is not None:
        sorted_keys = sorted_keys[:top_n]
    return [chunk_by_key[k] for k in sorted_keys]

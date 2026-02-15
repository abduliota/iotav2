"""
Extractive Answer Builder: copy-style extraction from retrieved chunks (no paraphrasing).
Used for fact_definition, metadata, and optionally procedural intents.
Returns contiguous span(s) from chunk text with mandatory (Page X) citation.
"""
from __future__ import annotations

import re
from typing import Any


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokens (min 2 chars) for overlap scoring."""
    tokens = set(re.findall(r"[a-z0-9\u0600-\u06ff]{2,}", (text or "").lower()))
    return tokens


# Header-like patterns: titles, article/section labels, short all-caps lines
_HEADER_PATTERNS = re.compile(
    r"^(?:Article\s+\d+|Section\s+\d+|المادة\s+\d+|تعريفات|Definitions|"
    r"BANKING\s+CONTROL\s+LAW|CHAPTER\s+\d+|Part\s+\d+[IVX]*)\s*[.:]?\s*$",
    re.IGNORECASE,
)


def _is_header_like(sentence: str, min_content_words: int = 8) -> bool:
    """True if sentence looks like a title/header: short, all-caps, or matches header patterns."""
    s = (sentence or "").strip()
    if not s or len(s) < 20:
        return True
    words = re.findall(r"[a-z0-9\u0600-\u06ff]{2,}", s, re.IGNORECASE)
    if len(words) < min_content_words and not any(w.islower() for w in words if len(w) > 2):
        return True
    if _HEADER_PATTERNS.match(s):
        return True
    # Mostly all-caps (Latin)
    caps = sum(1 for c in s if c.isupper() and ord(c) < 128)
    letters = sum(1 for c in s if c.isalpha() and ord(c) < 128)
    if letters and caps / letters > 0.7:
        return True
    return False


def _sentences_from_content(content: str) -> list[str]:
    """Split content into sentences (period, newline, question mark, etc.)."""
    if not content or not content.strip():
        return []
    text = content.strip()
    # Split on sentence boundaries; keep delimiter attached to previous
    parts = re.split(r"(?<=[.!?\n])\s+", text)
    return [p.strip() for p in parts if p.strip() and len(p.strip()) >= 10]


def _score_sentence(sentence: str, query_tokens: set[str]) -> tuple[int, bool]:
    """
    Returns (overlap_score, is_content_like).
    Overlap = number of query tokens in sentence.
    Deprioritize header-like or very short sentences via is_content_like.
    """
    sent_tokens = _tokenize(sentence)
    overlap = len(query_tokens & sent_tokens)
    content_like = not _is_header_like(sentence) and len(sent_tokens) >= 8
    return (overlap, content_like)


def _best_span_from_chunk(content: str, query: str, max_sentences: int = 3) -> str:
    """
    Return the best contiguous span (1--max_sentences) from content that matches the query.
    Prefers real content over title/header sentences; if best span is still header-like, uses next-best.
    """
    sentences = _sentences_from_content(content)
    if not sentences:
        return (content or "")[:400].strip() or ""
    query_tokens = _tokenize(query)
    if not query_tokens:
        # Prefer first non-header sentence
        for s in sentences:
            if not _is_header_like(s):
                return s[:400].strip()
        return sentences[0][:400].strip() if sentences else ""

    scored = []
    for i, s in enumerate(sentences):
        overlap, content_like = _score_sentence(s, query_tokens)
        # Strongly prefer content-like: add bonus so content sentences rank above headers
        effective = overlap + (1000 if content_like else 0)
        scored.append((effective, overlap, content_like, i, s))

    # Build candidate spans: consecutive runs, sorted by total effective score (then by content_like)
    candidates: list[tuple[int, int, bool, int, list[str]]] = []
    for start in range(len(sentences)):
        for length in range(1, min(max_sentences + 1, len(sentences) - start + 1)):
            run = sentences[start : start + length]
            total_eff = sum(scored[j][0] for j in range(start, start + length))
            total_overlap = sum(scored[j][1] for j in range(start, start + length))
            any_content = any(scored[j][2] for j in range(start, start + length))
            word_count = sum(len(re.findall(r"[a-z0-9\u0600-\u06ff]{2,}", sent, re.IGNORECASE)) for sent in run)
            candidates.append((total_eff, total_overlap, any_content, word_count, run))

    # Sort: higher effective, then higher overlap, then content-like, then more words
    candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)

    # Prefer first candidate that is real content (not header-like)
    for _eff, _overlap, any_content, word_count, span_sentences in candidates:
        result = " ".join(span_sentences).strip()
        if len(result) > 600:
            result = result[:600].strip()
        if word_count >= 15 and not _is_header_like(result, min_content_words=5):
            return result

    # Fallback: return best span even if header-like
    if candidates:
        span_sentences = candidates[0][4]
        result = " ".join(span_sentences).strip()
        return result[:600].strip() if len(result) > 600 else result
    result = " ".join(sentences[:max_sentences]).strip()
    return result[:600].strip() if len(result) > 600 else result


def build_extractive_answer(
    query: str,
    chunks: list[dict[str, Any]],
    intent: str,
    *,
    max_chunks: int = 1,
) -> str:
    """
    Build an answer by extracting the best span from the top chunk(s). No LLM.
    Returns "extracted span (Page X)." so citation is mandatory.
    For synthesis intent, returns empty string (caller should use generative path).
    """
    if not query or not chunks:
        return ""
    # Single-document default: use top max_chunks (1 for fact_definition/metadata)
    use = chunks[:max_chunks]
    best_chunk = use[0]
    content = (best_chunk.get("content") or "").strip()
    if not content:
        return ""
    page = best_chunk.get("page_start") or best_chunk.get("page_end") or 1
    try:
        page = int(page)
    except (TypeError, ValueError):
        page = 1
    span = _best_span_from_chunk(content, query, max_sentences=3)
    if not span:
        span = content[:400].strip()
    # Ensure citation is always present
    span = span.rstrip(".")
    return f"{span} (Page {page})."

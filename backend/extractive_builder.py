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


def _sentences_from_content(content: str) -> list[str]:
    """Split content into sentences (period, newline, question mark, etc.)."""
    if not content or not content.strip():
        return []
    text = content.strip()
    # Split on sentence boundaries; keep delimiter attached to previous
    parts = re.split(r"(?<=[.!?\n])\s+", text)
    return [p.strip() for p in parts if p.strip() and len(p.strip()) >= 10]


def _score_sentence(sentence: str, query_tokens: set[str]) -> int:
    """Number of query tokens that appear in sentence."""
    sent_tokens = _tokenize(sentence)
    return len(query_tokens & sent_tokens)


def _best_span_from_chunk(content: str, query: str, max_sentences: int = 3) -> str:
    """
    Return the best contiguous span (1--max_sentences) from content that matches the query.
    Uses token overlap; if no overlap, returns first sentence(s).
    """
    sentences = _sentences_from_content(content)
    if not sentences:
        return (content or "")[:400].strip() or ""
    query_tokens = _tokenize(query)
    if not query_tokens:
        return sentences[0][:400] if sentences else ""
    # Score each sentence; take best consecutive run up to max_sentences
    scored = [(_score_sentence(s, query_tokens), i, s) for i, s in enumerate(sentences)]
    best_start = 0
    best_total = sum(scored[i][0] for i in range(min(max_sentences, len(scored))))
    for start in range(len(sentences)):
        total = 0
        for j in range(start, min(start + max_sentences, len(sentences))):
            total += scored[j][0]
            if total > best_total:
                best_total = total
                best_start = start
        if start + max_sentences >= len(sentences):
            break
    span_sentences = sentences[best_start : best_start + max_sentences]
    result = " ".join(span_sentences)
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

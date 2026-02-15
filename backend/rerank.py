"""Rerank retrieved chunks with cross-encoder and optional Definitions section boost."""
from __future__ import annotations

import re
from typing import Any

_cross_encoder_model: Any = None

_DEFINITIONS_MARKERS = ("definitions", "definition", "تعريف", "تعريفات")


def _tokenize_for_title_match(text: str) -> set[str]:
    """Lowercase tokens (min 2 chars) for keyword overlap."""
    tokens = re.findall(r"[a-z0-9\u0600-\u06ff]{2,}", (text or "").lower())
    return set(tokens)


def _normalize_for_exact_match(text: str) -> str:
    """Normalize for exact/containment comparison: lowercase, collapse whitespace."""
    if not text:
        return ""
    return " ".join((text or "").lower().split())


def _title_exact_match_boost(query: str, chunk: dict[str, Any], boost: float) -> float:
    """Return boost if query (normalized) equals or is contained in section_title or document_name."""
    if not query or boost <= 0:
        return 0.0
    qn = _normalize_for_exact_match(query)
    if not qn:
        return 0.0
    title = _normalize_for_exact_match(chunk.get("section_title") or "")
    doc_name = _normalize_for_exact_match(chunk.get("document_name") or "")
    if title and (qn == title or qn in title or title in qn):
        return boost
    if doc_name and (qn == doc_name or qn in doc_name or doc_name in qn):
        return boost
    return 0.0


def _title_match_boost(query: str, chunk: dict[str, Any], min_match: int, boost: float) -> float:
    """Return boost if at least min_match query tokens appear in chunk's section_title."""
    if min_match < 1 or boost <= 0:
        return 0.0
    title = (chunk.get("section_title") or "").lower()
    if not title:
        return 0.0
    title_tokens = _tokenize_for_title_match(title)
    query_tokens = _tokenize_for_title_match(query or "")
    overlap = sum(1 for t in query_tokens if t in title_tokens)
    return boost if overlap >= min_match else 0.0


def _compute_dominant_document(chunks: list[dict[str, Any]]) -> str | None:
    """Return the document_name that appears most often in chunks (by chunk count)."""
    if not chunks:
        return None
    counts: dict[str, int] = {}
    for c in chunks:
        name = (c.get("document_name") or "").strip() or "(unknown)"
        counts[name] = counts.get(name, 0) + 1
    best_name: str | None = None
    best_count = 0
    for name, n in counts.items():
        if n > best_count:
            best_count = n
            best_name = name
    return best_name


def _apply_mmr_diversity(
    chunks_sorted: list[dict[str, Any]], top_n: int, lambda_param: float
) -> list[dict[str, Any]]:
    """Select top_n chunks with document diversity: prefer chunks from documents not yet in selection."""
    if not chunks_sorted or top_n <= 0:
        return chunks_sorted[:top_n] if top_n else chunks_sorted
    selected: list[dict[str, Any]] = []
    selected_docs: set[str] = set()
    used_indices: set[int] = set()
    for _ in range(min(top_n, len(chunks_sorted))):
        best_idx: int | None = None
        best_val = -1e9
        for i, c in enumerate(chunks_sorted):
            if i in used_indices:
                continue
            score = float(c.get("_rerank_score") or 0)
            doc = (c.get("document_name") or "").strip()
            diversity_bonus = 1.0 if doc not in selected_docs else 0.0
            val = lambda_param * score + (1.0 - lambda_param) * diversity_bonus
            if val > best_val:
                best_val = val
                best_idx = i
        if best_idx is None:
            break
        c = chunks_sorted[best_idx]
        selected.append(c)
        selected_docs.add((c.get("document_name") or "").strip())
        used_indices.add(best_idx)
    return selected


def _chunk_matches_preferred_docs(chunk: dict[str, Any], preferred: list[str]) -> bool:
    """True if chunk's document_name contains any preferred substring (case-insensitive)."""
    if not preferred:
        return False
    doc_name = (chunk.get("document_name") or "").lower()
    return any(hint.lower() in doc_name for hint in preferred)


def _get_cross_encoder():
    global _cross_encoder_model
    if _cross_encoder_model is None:
        from config import CROSS_ENCODER_MODEL
        from sentence_transformers import CrossEncoder
        _cross_encoder_model = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder_model


def _has_definitions_section(chunk: dict[str, Any]) -> bool:
    """True if chunk section_title or content suggests a Definitions section."""
    title = (chunk.get("section_title") or "").lower()
    content = (chunk.get("content") or "").lower()
    text = title + " " + content[:500]
    return any(m in text for m in _DEFINITIONS_MARKERS)


def _base_score(chunk: dict[str, Any]) -> float:
    """Existing similarity from retrieval (cosine_similarity or similarity)."""
    return float(
        chunk.get("cosine_similarity")
        or chunk.get("similarity")
        or chunk.get("score")
        or 0.0
    )


def rerank_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    *,
    use_cross_encoder: bool = True,
    definitions_boost: float = 0.15,
    top_n: int | None = None,
    preferred_doc_names: list[str] | None = None,
    intent: str | None = None,
) -> list[dict[str, Any]]:
    """
    Rerank chunks: optionally by cross-encoder (query, content), then add Definitions
    section boost, title match boost, document dominance boost, and keyword-document boost.
    Returns list sorted by score descending; if top_n set, return only first top_n.
    """
    if not chunks:
        return list(chunks)
    try:
        from config import (
            ENABLE_CROSS_ENCODER_RERANK,
            ENABLE_KEYWORD_DOCUMENT_BOOST,
            ENABLE_RERANKER_DOMINANCE_BOOST,
            ENABLE_RERANKER_TITLE_BOOST,
            ENABLE_RERANKER_TITLE_EXACT_BOOST,
            RERANKER_DEFINITIONS_BOOST,
            RERANKER_DOMINANT_DOC_BOOST,
            RERANKER_KEYWORD_DOCUMENT_BOOST,
            RERANKER_TITLE_KEYWORD_MIN_MATCH,
            RERANKER_TITLE_MATCH_BOOST,
            RERANKER_TITLE_EXACT_MATCH_BOOST,
        )
        use_cross_encoder = use_cross_encoder and ENABLE_CROSS_ENCODER_RERANK
        definitions_boost = RERANKER_DEFINITIONS_BOOST
    except ImportError:
        ENABLE_RERANKER_TITLE_BOOST = False
        ENABLE_RERANKER_TITLE_EXACT_BOOST = False
        ENABLE_RERANKER_DOMINANCE_BOOST = False
        ENABLE_KEYWORD_DOCUMENT_BOOST = False
        RERANKER_TITLE_MATCH_BOOST = 0.1
        RERANKER_TITLE_EXACT_MATCH_BOOST = 0.2
        RERANKER_TITLE_KEYWORD_MIN_MATCH = 1
        RERANKER_DOMINANT_DOC_BOOST = 0.05
        RERANKER_KEYWORD_DOCUMENT_BOOST = 0.12

    dominant_doc = _compute_dominant_document(chunks) if ENABLE_RERANKER_DOMINANCE_BOOST else None
    preferred = list(preferred_doc_names) if preferred_doc_names and ENABLE_KEYWORD_DOCUMENT_BOOST else []

    def _add_boosts(chunk: dict[str, Any], base: float) -> float:
        s = base
        if _has_definitions_section(chunk):
            s += definitions_boost
        if ENABLE_RERANKER_TITLE_BOOST and query:
            s += _title_match_boost(
                query, chunk, RERANKER_TITLE_KEYWORD_MIN_MATCH, RERANKER_TITLE_MATCH_BOOST
            )
        if ENABLE_RERANKER_TITLE_EXACT_BOOST and query:
            s += _title_exact_match_boost(query, chunk, RERANKER_TITLE_EXACT_MATCH_BOOST)
        if dominant_doc and (chunk.get("document_name") or "").strip() == dominant_doc:
            s += RERANKER_DOMINANT_DOC_BOOST
        if preferred and _chunk_matches_preferred_docs(chunk, preferred):
            s += RERANKER_KEYWORD_DOCUMENT_BOOST
        return s

    if use_cross_encoder and query:
        try:
            model = _get_cross_encoder()
            pairs = [(query, (c.get("content") or "")[:2000]) for c in chunks]
            scores = model.predict(pairs)
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
            else:
                scores = list(scores)
            for i, chunk in enumerate(chunks):
                s = float(scores[i]) if i < len(scores) else 0.0
                chunk["_rerank_score"] = _add_boosts(chunk, s)
        except Exception:
            for c in chunks:
                c["_rerank_score"] = _add_boosts(c, _base_score(c))
    else:
        for c in chunks:
            c["_rerank_score"] = _add_boosts(c, _base_score(c))
    chunks_sorted = sorted(chunks, key=lambda c: -(c.get("_rerank_score") or 0))
    try:
        from config import ENABLE_MMR_DIVERSITY, ENABLE_MMR_DIVERSITY_SYNTHESIS, RERANKER_MMR_DIVERSITY_LAMBDA
        use_mmr = ENABLE_MMR_DIVERSITY or (intent == "synthesis" and ENABLE_MMR_DIVERSITY_SYNTHESIS)
        if use_mmr and top_n is not None and len(chunks_sorted) > 1:
            chunks_sorted = _apply_mmr_diversity(chunks_sorted, top_n, RERANKER_MMR_DIVERSITY_LAMBDA)
    except ImportError:
        pass
    if top_n is not None:
        chunks_sorted = chunks_sorted[:top_n]
    # Keep _rerank_score on chunks for similarity-gate bypass in simple_rag (caller pops after use)
    return chunks_sorted

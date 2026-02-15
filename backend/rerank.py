"""Rerank retrieved chunks with cross-encoder and optional Definitions section boost."""
from __future__ import annotations

from typing import Any

_cross_encoder_model: Any = None

_DEFINITIONS_MARKERS = ("definitions", "definition", "تعريف", "تعريفات")


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
) -> list[dict[str, Any]]:
    """
    Rerank chunks: optionally by cross-encoder (query, content), then add Definitions
    section boost. If cross-encoder disabled, use existing similarity + boost.
    Returns list sorted by score descending; if top_n set, return only first top_n.
    """
    if not chunks:
        return list(chunks)
    try:
        from config import (
            ENABLE_CROSS_ENCODER_RERANK,
            RERANKER_DEFINITIONS_BOOST,
        )
        use_cross_encoder = use_cross_encoder and ENABLE_CROSS_ENCODER_RERANK
        definitions_boost = RERANKER_DEFINITIONS_BOOST
    except ImportError:
        pass
    boost = definitions_boost
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
                if _has_definitions_section(chunk):
                    s += boost
                chunk["_rerank_score"] = s
        except Exception:
            for c in chunks:
                c["_rerank_score"] = _base_score(c) + (boost if _has_definitions_section(c) else 0)
    else:
        for c in chunks:
            c["_rerank_score"] = _base_score(c) + (boost if _has_definitions_section(c) else 0)
    chunks_sorted = sorted(chunks, key=lambda c: -(c.get("_rerank_score") or 0))
    for c in chunks_sorted:
        c.pop("_rerank_score", None)
    if top_n is not None:
        chunks_sorted = chunks_sorted[:top_n]
    return chunks_sorted

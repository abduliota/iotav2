"""Semantic grounding: answer vs chunk embeddings, token overlap, intent-based thresholds."""
from __future__ import annotations

import re
from typing import Any

try:
    from config import (
        GROUNDING_SOFT_FAIL_THRESHOLD,
        GROUNDING_HARD_FAIL_THRESHOLD,
        GROUNDING_PARTIAL_BAND,
        SEMANTIC_GROUNDING_THRESHOLD_FACT_DEFINITION,
        SEMANTIC_GROUNDING_THRESHOLD_METADATA,
        SEMANTIC_GROUNDING_THRESHOLD_SYNTHESIS,
        SEMANTIC_GROUNDING_THRESHOLD_OTHER,
        USE_SEMANTIC_GROUNDING,
    )
except ImportError:
    USE_SEMANTIC_GROUNDING = False
    GROUNDING_SOFT_FAIL_THRESHOLD = 0.5
    GROUNDING_HARD_FAIL_THRESHOLD = 0.35
    GROUNDING_PARTIAL_BAND = True
    SEMANTIC_GROUNDING_THRESHOLD_FACT_DEFINITION = 0.4
    SEMANTIC_GROUNDING_THRESHOLD_METADATA = 0.4
    SEMANTIC_GROUNDING_THRESHOLD_SYNTHESIS = 0.45
    SEMANTIC_GROUNDING_THRESHOLD_OTHER = 0.45


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return (dot / (na * nb)) if (na and nb) else 0.0


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", (text or "").lower()))


def token_overlap_ratio(answer: str, context: str) -> float:
    """Fraction of answer tokens that appear in context (0–1)."""
    a_tokens = _tokenize(answer)
    c_tokens = _tokenize(context)
    if not a_tokens:
        return 1.0
    return len(a_tokens & c_tokens) / len(a_tokens)


def semantic_grounding_score(
    answer: str,
    chunks: list[dict[str, Any]],
    *,
    embed_query_fn: Any = None,
    embed_chunk_fn: Any = None,
    max_chunks: int = 10,
) -> float:
    """
    Max cosine similarity between answer embedding and chunk content embeddings.
    Returns value in [0, 1] (assuming cosine in [-1,1], we clamp to [0,1] for readability).
    """
    if not answer or not chunks:
        return 0.0
    try:
        from embeddings import embed_chunk as _embed_chunk
        from embeddings import embed_query as _embed_query
        embed_query_fn = embed_query_fn or _embed_query
        embed_chunk_fn = embed_chunk_fn or _embed_chunk
    except ImportError:
        return 0.0
    try:
        body = re.sub(r"\(Page\s+\d+\)|\(Pages\s+\d+\s*[–\-]\s*\d+\)", "", answer, flags=re.IGNORECASE)
        body = re.sub(r"\(Source:\s*provided context\.?\)", "", body, flags=re.IGNORECASE).strip()[:3000]
        if not body:
            return 0.0
        q_emb = embed_query_fn(body)
        best = -1.0
        for c in chunks[:max_chunks]:
            content = (c.get("content") or "").strip()[:2000]
            if not content:
                continue
            c_emb = embed_chunk_fn(content)
            sim = _cosine(q_emb, c_emb)
            if sim > best:
                best = sim
        return max(0.0, min(1.0, (best + 1) / 2.0))  # map [-1,1] -> [0,1] for thresholding
    except Exception:
        return 0.0


def get_grounding_threshold_for_intent(intent: str) -> float:
    try:
        from config import (
            SEMANTIC_GROUNDING_THRESHOLD_FACT_DEFINITION,
            SEMANTIC_GROUNDING_THRESHOLD_METADATA,
            SEMANTIC_GROUNDING_THRESHOLD_OTHER,
            SEMANTIC_GROUNDING_THRESHOLD_SYNTHESIS,
        )
        if intent == "fact_definition":
            return SEMANTIC_GROUNDING_THRESHOLD_FACT_DEFINITION
        if intent == "metadata":
            return SEMANTIC_GROUNDING_THRESHOLD_METADATA
        if intent == "synthesis":
            return SEMANTIC_GROUNDING_THRESHOLD_SYNTHESIS
        return SEMANTIC_GROUNDING_THRESHOLD_OTHER
    except ImportError:
        return 0.4


def grounding_decision(
    answer: str,
    context_text: str,
    chunks: list[dict[str, Any]],
    intent: str,
) -> tuple[str, float, str]:
    """
    Returns (decision, score, message) where:
    - decision: "pass" | "soft_fail" | "hard_fail"
    - score: semantic score in [0,1]
    - message: optional uncertainty phrase for soft_fail
    """
    score = 0.0
    if USE_SEMANTIC_GROUNDING and answer and chunks:
        score = semantic_grounding_score(answer, chunks)
        threshold = get_grounding_threshold_for_intent(intent)
        overlap = token_overlap_ratio(answer, context_text)
        combined = 0.7 * score + 0.3 * overlap
        if combined >= GROUNDING_SOFT_FAIL_THRESHOLD:
            return ("pass", score, "")
        if combined < GROUNDING_HARD_FAIL_THRESHOLD:
            return ("hard_fail", score, "")
        if GROUNDING_PARTIAL_BAND:
            return ("soft_fail", score, " (Source: provided context).")
        return ("hard_fail", score, "")
    return ("pass", 0.0, "")


def uncertainty_phrase(score: float) -> str:
    """Optional phrase to append when score is in partial band."""
    try:
        from config import SIMPLE_RAG_UNCERTAINTY_PHRASE
        return SIMPLE_RAG_UNCERTAINTY_PHRASE or ""
    except ImportError:
        return ""

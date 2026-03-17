from __future__ import annotations

"""
Thin embedding wrapper for the IOTAV3 RAG pipeline.

For now this delegates to the existing embedding implementation in the
root backend to avoid duplicating model loading code, but keeps the
import surface small and IOTAV3‑specific.
"""

from typing import List

from embeddings import embed_query as _embed_query  # type: ignore


def embed_query(text: str) -> List[float]:
    """
    Generate an embedding vector for the given query text.
    """

    return _embed_query(text)


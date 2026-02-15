"""OpenAI embeddings for Stage 3/4. Generates fixed-dimension vectors."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from config import AZURE_EMBEDDING_DIMENSION, AZURE_EMBEDDING_MODEL

if TYPE_CHECKING:
    from openai import OpenAI

_client: "OpenAI | None" = None


def _get_client() -> "OpenAI":
    """Lazy-load OpenAI client. Requires OPENAI_API_KEY env var."""
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI()
    return _client


def normalize_text(text: str) -> str:
    """Remove excessive whitespace for embedding input."""
    if not text:
        return ""
    # Collapse multiple spaces/newlines to single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def embed_chunk(content: str) -> list[float]:
    """
    Generate embedding for chunk content.
    Returns 3072-dim vector as list[float].
    """
    normalized = normalize_text(content)
    input_text = normalized

    client = _get_client()
    response = client.embeddings.create(
        model=AZURE_EMBEDDING_MODEL,
        input=input_text,
        dimensions=AZURE_EMBEDDING_DIMENSION,
    )

    embedding = response.data[0].embedding
    if len(embedding) != AZURE_EMBEDDING_DIMENSION:
        raise ValueError(
            f"Expected {AZURE_EMBEDDING_DIMENSION} dimensions, got {len(embedding)}"
        )
    return embedding


def embed_query(query: str) -> list[float]:
    """
    Generate embedding for user query.
    Returns 3072-dim vector as list[float].
    """
    normalized = normalize_text(query)
    input_text = normalized

    client = _get_client()
    response = client.embeddings.create(
        model=AZURE_EMBEDDING_MODEL,
        input=input_text,
        dimensions=AZURE_EMBEDDING_DIMENSION,
    )

    embedding = response.data[0].embedding
    if len(embedding) != AZURE_EMBEDDING_DIMENSION:
        raise ValueError(
            f"Expected {AZURE_EMBEDDING_DIMENSION} dimensions, got {len(embedding)}"
        )
    return embedding

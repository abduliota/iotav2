"""Embeddings for Stage 3/4: OpenAI or multilingual (e5/bge-m3). Generates fixed-dimension vectors."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from config import (
    AZURE_EMBEDDING_DIMENSION,
    AZURE_EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    MULTILINGUAL_EMBEDDING_DIMENSION,
    MULTILINGUAL_EMBEDDING_MODEL,
    USE_MULTILINGUAL_EMBEDDING,
)

if TYPE_CHECKING:
    from openai import OpenAI

_openai_client: "OpenAI | None" = None
_multilingual_model: "object | None" = None


def _get_openai_client() -> "OpenAI":
    """Lazy-load OpenAI client. Requires OPENAI_API_KEY env var."""
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI()
    return _openai_client


def _get_multilingual_model() -> "object":
    """Lazy-load SentenceTransformer model for multilingual embeddings."""
    global _multilingual_model
    if _multilingual_model is None:
        from sentence_transformers import SentenceTransformer
        _multilingual_model = SentenceTransformer(MULTILINGUAL_EMBEDDING_MODEL)
    return _multilingual_model


def normalize_text(text: str) -> str:
    """Remove excessive whitespace for embedding input."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _embed_openai_chunk(content: str) -> list[float]:
    client = _get_openai_client()
    response = client.embeddings.create(
        model=AZURE_EMBEDDING_MODEL,
        input=content,
        dimensions=AZURE_EMBEDDING_DIMENSION,
    )
    embedding = response.data[0].embedding
    if len(embedding) != AZURE_EMBEDDING_DIMENSION:
        raise ValueError(
            f"Expected {AZURE_EMBEDDING_DIMENSION} dimensions, got {len(embedding)}"
        )
    return embedding


def _embed_openai_query(query: str) -> list[float]:
    client = _get_openai_client()
    response = client.embeddings.create(
        model=AZURE_EMBEDDING_MODEL,
        input=query,
        dimensions=AZURE_EMBEDDING_DIMENSION,
    )
    embedding = response.data[0].embedding
    if len(embedding) != AZURE_EMBEDDING_DIMENSION:
        raise ValueError(
            f"Expected {AZURE_EMBEDDING_DIMENSION} dimensions, got {len(embedding)}"
        )
    return embedding


def _embed_multilingual_chunk(content: str) -> list[float]:
    model = _get_multilingual_model()
    # e5 and bge-m3 use "passage" prefix for documents
    prefixed = f"passage: {content}" if content else "passage: "
    vec = model.encode(prefixed, convert_to_numpy=True)
    embedding = vec.tolist()
    if len(embedding) != MULTILINGUAL_EMBEDDING_DIMENSION:
        raise ValueError(
            f"Expected {MULTILINGUAL_EMBEDDING_DIMENSION} dimensions, got {len(embedding)}"
        )
    return embedding


def _embed_multilingual_query(query: str) -> list[float]:
    model = _get_multilingual_model()
    prefixed = f"query: {query}" if query else "query: "
    vec = model.encode(prefixed, convert_to_numpy=True)
    embedding = vec.tolist()
    if len(embedding) != MULTILINGUAL_EMBEDDING_DIMENSION:
        raise ValueError(
            f"Expected {MULTILINGUAL_EMBEDDING_DIMENSION} dimensions, got {len(embedding)}"
        )
    return embedding


def embed_chunk(content: str) -> list[float]:
    """
    Generate embedding for chunk content.
    Returns vector of length EMBEDDING_DIMENSION (OpenAI or multilingual per config).
    """
    normalized = normalize_text(content)
    if not normalized:
        normalized = " "
    if USE_MULTILINGUAL_EMBEDDING:
        return _embed_multilingual_chunk(normalized)
    return _embed_openai_chunk(normalized)


def embed_query(query: str) -> list[float]:
    """
    Generate embedding for user query.
    Returns vector of length EMBEDDING_DIMENSION (OpenAI or multilingual per config).
    """
    normalized = normalize_text(query)
    if not normalized:
        normalized = " "
    if USE_MULTILINGUAL_EMBEDDING:
        return _embed_multilingual_query(normalized)
    return _embed_openai_query(normalized)

---
title: Embedding Model for Memory Items
status: accepted
date: 2026-03-10
---

## Context

The system already uses vector embeddings for:

- Regulatory document chunks (stored in Supabase with pgvector).
- Query embeddings for retrieval and grounding.

Embeddings are configured via `backend/config.py`:

- `USE_MULTILINGUAL_EMBEDDING` (default `True`).
- `MULTILINGUAL_EMBEDDING_MODEL` (default `intfloat/multilingual-e5-small`).
- `MULTILINGUAL_EMBEDDING_DIMENSION` (default `384`).
- `AZURE_EMBEDDING_MODEL` and `AZURE_EMBEDDING_DIMENSION` (for optional Azure integration).
- Effective dimension: `EMBEDDING_DIMENSION`.

We now store episodic memory items (`memory_item`) and want to query them
semantically. We must decide how to embed these items and what vector dimension
to use in the `memory_item_embedding` table.

## Decision

We reuse the **same embedding pipeline and dimension** for memory items as we do
for regulatory document chunks:

- `embed_chunk` / `embed_query` in `backend/embeddings.py` generate vectors
  that have length `EMBEDDING_DIMENSION`.
- `memory_item_embedding.embedding` is defined as `vector(EMBEDDING_DIMENSION)`
  in SQL.

This ensures:

- A single, consistent embedding configuration across:
  - Document chunks.
  - User queries.
  - Episodic memory items.
- Simplified maintenance: when we change embedding backend or dimension, there is
  only one conceptual embedding space to reason about.

Migrations:

- `003_memory_tables.sql`:
  - Creates `memory_item_embedding` with `vector(384)` by default and a comment
    indicating that this must be kept in sync with `EMBEDDING_DIMENSION`.
- `004_memory_match_rpc.sql`:
  - Defines `match_memory_items(query_embedding vector(384), ...)` using the
    same dimension.

Runtime checks:

- `embed_memory_item` in `backend/memory.py`:
  - Computes `vec = embed_chunk(text)`.
  - If `len(vec) != EMBEDDING_DIMENSION`, it **skips** inserting the embedding
    (fails soft) to avoid corrupting the index.

## Consequences

### Positive

- **Consistency**: All semantic operations (retrieval, similarity, grounding)
  operate in the same vector space.
- **Reduced complexity**: No need to maintain or reason about multiple
  embedding models or dimensions for different data types.
- **Reusability**: If we later want to perform hybrid retrieval that considers
  both document chunks and memory items jointly, we already have a shared
  embedding space.

### Negative / Trade-offs

- **Coupled migrations**: Changing `EMBEDDING_DIMENSION` (e.g. switching to a
  new embedding model) requires:
  - Updating all `vector(…)` columns (documents and memory tables).
  - Re-embedding both corpus and memory items.
- **Model lock-in**: We are effectively standardizing on a single embedding
  model for all semantic search use cases (documents + memory).

## Migration / Re-Embedding Plan

If we later decide to change the embedding model or dimension:

1. Update `MULTILINGUAL_EMBEDDING_MODEL`, `MULTILINGUAL_EMBEDDING_DIMENSION`,
   and/or Azure configuration in `config.py`, ensuring `EMBEDDING_DIMENSION`
   reflects the new value.
2. Create a new SQL migration to:
   - Drop or alter `embedding` columns on:
     - The chunks embeddings table (e.g. `sama_nora_chunks`).
     - `memory_item_embedding`.
   - Re-create them with the new dimension (`vector(NEW_DIM)`).
3. Run a re-embedding script for:
   - All regulatory chunks.
   - All existing `memory_item` rows, populating `memory_item_embedding`.
4. Monitor:
   - Retrieval quality.
   - Memory search quality.
   - Performance impact (index size, query latency).

Until such a migration is executed, both document and memory embeddings must
continue to use the current `EMBEDDING_DIMENSION` configuration.


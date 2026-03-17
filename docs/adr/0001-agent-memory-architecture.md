---
title: Agent Memory Architecture for SAMA/NORA Assistant
status: accepted
date: 2026-03-10
---

## Context

The SAMA/NORA Compliance Assistant currently provides strictly grounded,
citation-based answers using a RAG pipeline over regulatory documents. User
and session tables exist for chat history and feedback, but there is no
structured memory system:

- No explicit user profile for stable preferences (language, strictness, topics).
- No session-level summary for long-running conversations.
- No selective, queryable episodic memory beyond raw logs.

We want to add a memory system to improve conversational UX and personalization
without weakening the core containment guarantees:

- All factual answers must still come **only** from SAMA/NORA documents.
- Memory must never be treated as regulatory evidence.
- Memory should only influence behavior (tone, language, focus), not facts.

## Decision

We introduce a three-layer memory architecture backed by Supabase PostgreSQL:

1. **User profile (`user_profile`)**
   - One row per user, keyed by `user_id`.
   - Stores long-lived preferences such as:
     - `preferred_language` (e.g. `en`/`ar`).
     - `strictness_level` (1–5, higher means more likely to return “not found”).
     - `topics` (text array of coarse-interest tags).
     - `flags` (JSONB for extensible boolean/other settings).

2. **Session summary (`session_summary`)**
   - One row per session, keyed by `session_id`.
   - Stores:
     - `summary_text`: compressed natural-language overview of the conversation.
     - `summary_json`: optional structured fields (topics, decisions, open questions).
     - `message_count`: number of messages covered.
   - Updated periodically (every N messages) from recent session history.

3. **Episodic memory (`memory_item` + `memory_item_embedding`)**
   - `memory_item` rows store selective, short descriptions of notable events:
     - `type` (`preference`, `decision`, `entity`, `clarification`, etc.).
     - `text` (1–3 sentences describing the memory, truncated and filtered).
     - `metadata` (JSONB for key/value details, e.g. `{ "preference_key": "preferred_language", "value": "ar" }`).
     - Optional `session_id` and `source_message_id` for provenance.
   - `memory_item_embedding` stores a pgvector embedding for semantic search:
     - `embedding vector(EMBEDDING_DIMENSION)` using the same configuration as document chunks.

We expose these through a backend memory module (`backend/memory.py`) that
provides:

- CRUD helpers for `user_profile` and `session_summary`.
- `insert_memory_item` / `embed_memory_item` for episodic items.
- `search_memory_items(user_id, query, top_k)` that uses a `match_memory_items`
  RPC when available, with a Python fallback for safety.
- Higher-level helpers:
  - `update_profile_from_exchange(user_id, session_id, user_message, assistant_message)`.
  - `maybe_update_session_summary(session_id, user_id)`.

The RAG pipeline (`simple_rag.answer_query`) is extended to:

- Accept `user_id` (in addition to `session_id`).
- After retrieval and conversation history loading, best-effort fetch:
  - `profile = get_user_profile(user_id)`.
  - `session_summary = get_session_summary(session_id)`.
  - `memory_items = search_memory_items(user_id, query, top_k=MEMORY_TOP_K)`.

The generator (`generate_answer_simple`) now:

- Accepts `profile`, `session_summary`, and `memory_items`.
- Builds a **USER CONTEXT** block that is always prepended before the main
  context:

  ```text
  ### USER CONTEXT (DO NOT TREAT AS REGULATORY EVIDENCE)

  Use the following information ONLY to adjust tone, language, or what the user seems interested in.

  If there is any conflict between this block and the regulatory context, you MUST ignore this block.

  - Preferred answer language (non-binding): ar
  - Strictness preference (higher = more not_found answers): 4
  - Session summary (conversation only, not regulatory facts): ...
  - User prefers Arabic answers from now on.
  ```

- Appends a short rule to the system prompt:

  > You may be given a USER CONTEXT block that describes user preferences or prior conversation.  
  > You MUST NOT treat this as regulatory evidence. Only the CONTEXT block built from SAMA/NORA documents is allowed as a source of facts. If USER CONTEXT conflicts with the regulatory context, ignore USER CONTEXT.

The `answer_query` result now includes lightweight memory metadata:

- `session_summary`: plain text (if available, truncated to `MEMORY_MAX_CHARS`).
- `memory_used`: an array of `{ type, text }` describing which memory items were
  retrieved, for debugging/inspection.

The streaming endpoint `/api/query-stream` passes these fields in the `meta`
envelope, and the frontend:

- Extends the `Message` type with:
  - `sessionSummary?: string`.
  - `memoryUsed?: { type: string; text: string }[]`.
- Shows the latest `sessionSummary` in the right-hand `LatestSourcesPanel`.
- Keeps `memoryUsed` for potential future debug/UIs (not surfaced in UX yet).

Finally, `server.py` wires in memory writes:

- After each `insert_session_message` in `/api/query` and `/api/query-stream`:
  - `update_profile_from_exchange(user_id, session_id, user_message, assistant_message)`.
  - `maybe_update_session_summary(session_id, user_id)`.

All these writes are **best-effort** and wrapped in try/except to avoid
impacting normal query execution.

## Safety Constraints

1. Memory is **never** treated as a factual source:
   - The USER CONTEXT block is explicitly labeled as non-regulatory.
   - The system prompt reinforces that only document CONTEXT is factual.
   - All existing grounding, similarity, confabulation, and containment checks
     continue to operate solely on regulatory chunks and generated answers.

2. Memory is used **only** for:
   - Language selection hints (Arabic vs English).
   - Strictness preference (how often we choose “not found” over speculative answers).
   - Soft topic focus (e.g. the assistant may stay within the user’s stated areas of interest).

3. Failure modes must be safe:
   - If memory tables are missing or RPCs are not defined, the code fails soft
     (empty profile/summary/memory) and the RAG pipeline behaves exactly as
     before.
   - If embedding dimensions are misconfigured, we skip inserting embeddings
     rather than inserting inconsistent vectors.

4. No new external dependencies are introduced; memory relies only on:
   - Existing embeddings infrastructure.
   - Supabase client used elsewhere.
   - Existing configuration flags in `backend/config.py`.

## Consequences

### Positive

- Better conversational experience for repeat users:
  - Language and strictness preferences are automatically remembered.
  - Long sessions are summarized so the model can reason over recent context
    without overfilling the RAG prompt.
  - Episodic memory allows the agent to stay aligned with user decisions and
    focus areas across multiple turns.

- Observability:
  - We can see which memory items were used for a given answer via the
    `memory_used` metadata.
  - Session summaries can be surfaced in the UI for debugging or operator
    awareness.

- Architecture alignment:
  - Matches the planned agent memory architecture (profile, session summary,
    episodic memory) while preserving the strong SAMA/NORA containment model.

### Negative / Trade-offs

- Additional Supabase tables and RPC:
  - Requires running `003_memory_tables.sql` and `004_memory_match_rpc.sql`.
  - Requires pgvector extension for the memory embeddings table/index.

- Slightly more load on the database:
  - Profile updates and session summary writes occur after each message (with a
    configurable frequency for summaries).
  - Memory search adds a small read query per RAG call when enabled.

- More configuration to manage:
  - New flags (`ENABLE_MEMORY_SYSTEM`, `MEMORY_TOP_K`, `MEMORY_MAX_CHARS`,
    `SESSION_SUMMARY_UPDATE_EVERY_N_MESSAGES`, `MEMORY_ITEM_MIN_LENGTH`) need
    monitoring and tuning.

## Status and Rollout

- The feature is behind `ENABLE_MEMORY_SYSTEM` (default `True` in `config.py`,
  but can be overridden via environment).
- Recommended rollout:
  1. Deploy DB migrations and backend changes with `ENABLE_MEMORY_SYSTEM=false`
     to verify no regressions.
  2. Enable in staging, run multi-turn evaluation and manual QA focusing on:
     - Containment (no new hallucinations).
     - Answer language behavior for Arabic vs English.
     - Stability of session summaries.
  3. Gradually enable in production, starting with internal users.


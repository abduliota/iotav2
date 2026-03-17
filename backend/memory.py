"""Memory subsystem: user profiles, session summaries, and episodic memory items.

This module provides a thin service layer around Supabase for:
- user_profile: long-lived structured preferences
- session_summary: rolling summaries of a session's conversation
- memory_item + memory_item_embedding: selective episodic memory with embeddings

All operations are best-effort and should fail soft: callers must not depend on
memory being available for correctness of the core RAG pipeline.
"""
from __future__ import annotations

from typing import Any, Iterable

from config import (
    ENABLE_EPISODIC_MEMORY_WRITES,
    ENABLE_MEMORY_SYSTEM,
    EMBEDDING_DIMENSION,
    MEMORY_ITEM_MIN_LENGTH,
    MEMORY_MAX_CHARS,
    MEMORY_TOP_K,
    SESSION_SUMMARY_UPDATE_EVERY_N_MESSAGES,
)
from embeddings import embed_chunk, embed_query
from supabase_client import get_client
from users_sessions import get_session_message_history


def get_user_profile(user_id: str) -> dict[str, Any] | None:
    """Return user_profile row for user_id or None if not found."""
    client = get_client()
    result = (
        client.table("user_profile")
        .select("*")
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    rows = result.data or []
    return rows[0] if rows else None


def upsert_user_profile(user_id: str, **fields: Any) -> None:
    """Upsert user_profile for the given user_id with provided fields."""
    if not user_id:
        return
    data: dict[str, Any] = {"user_id": user_id}
    data.update({k: v for k, v in fields.items() if v is not None})
    client = get_client()
    client.table("user_profile").upsert(data).execute()


def get_session_summary(session_id: str) -> dict[str, Any] | None:
    """Return session_summary row for session_id or None if not found."""
    if not session_id:
        return None
    client = get_client()
    result = (
        client.table("session_summary")
        .select("*")
        .eq("session_id", session_id)
        .limit(1)
        .execute()
    )
    rows = result.data or []
    return rows[0] if rows else None


def upsert_session_summary(
    session_id: str,
    user_id: str,
    summary_text: str,
    summary_json: dict[str, Any] | None,
    message_count: int,
) -> None:
    """Upsert session_summary metadata for a session."""
    if not session_id or not user_id:
        return
    client = get_client()
    data: dict[str, Any] = {
        "session_id": session_id,
        "user_id": user_id,
        "summary_text": summary_text,
        "message_count": int(message_count),
    }
    if summary_json is not None:
        data["summary_json"] = summary_json
    client.table("session_summary").upsert(data).execute()


def insert_memory_item(
    user_id: str,
    session_id: str | None,
    type_: str,
    text: str,
    metadata: dict[str, Any] | None = None,
    source_message_id: str | None = None,
) -> str | None:
    """Insert a new episodic memory item and return its id.

    Returns None if insertion fails or text is too short.
    """
    if not user_id or not type_:
        return None
    normalized = (text or "").strip()
    if len(normalized) < MEMORY_ITEM_MIN_LENGTH:
        return None
    client = get_client()
    payload: dict[str, Any] = {
        "user_id": user_id,
        "type": type_,
        "text": normalized[: MEMORY_MAX_CHARS],
    }
    if session_id:
        payload["session_id"] = session_id
    if metadata:
        payload["metadata"] = metadata
    if source_message_id:
        payload["source_message_id"] = source_message_id
    result = client.table("memory_item").insert(payload).execute()
    rows = result.data or []
    if not rows:
        return None
    return rows[0].get("id")


def embed_memory_item(memory_item_id: str, text: str) -> None:
    """Create or replace embedding row for a memory_item.

    Uses the same embedding pipeline as document chunks to ensure the
    dimension matches EMBEDDING_DIMENSION.
    """
    if not memory_item_id:
        return
    content = (text or "").strip()
    if not content:
        return
    vec = embed_chunk(content)
    # Guard against dimension mismatch at runtime
    if len(vec) != EMBEDDING_DIMENSION:
        # Fail soft: skip inserting if configuration and DB are inconsistent.
        return
    client = get_client()
    upsert_payload = {
        "memory_item_id": memory_item_id,
        "embedding": vec,
    }
    client.table("memory_item_embedding").upsert(upsert_payload).execute()


def search_memory_items(user_id: str, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """Return top_k memory items for a user by semantic similarity to query.

    Tries the match_memory_items RPC when available, falling back to a
    simple in-Python cosine similarity over a limited set of items.
    """
    if not user_id or not query or not query.strip():
        return []
    if top_k is None:
        top_k = MEMORY_TOP_K
    q_vec = embed_query(query)
    if len(q_vec) != EMBEDDING_DIMENSION:
        return []
    client = get_client()
    try:
        # Prefer RPC if configured in the database
        result = client.rpc(
            "match_memory_items",
            {
                "query_embedding": q_vec,
                "match_count": top_k,
                "match_user_id": user_id,
            },
        ).execute()
        rows: list[dict[str, Any]] = result.data or []
        return rows
    except Exception:
        # Fallback: pull a limited set of memory items for the user and compute cosine similarity in Python.
        try:
            base = (
                client.table("memory_item")
                .select("id, user_id, session_id, type, text, metadata, created_at")
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .limit(50)
                .execute()
            )
            items: list[dict[str, Any]] = base.data or []
            if not items:
                return []
            # Fetch corresponding embeddings
            ids = [row["id"] for row in items if row.get("id")]
            if not ids:
                return []
            emb_result = (
                client.table("memory_item_embedding")
                .select("memory_item_id, embedding")
                .in_("memory_item_id", ids)
                .execute()
            )
            emb_map = {row["memory_item_id"]: row["embedding"] for row in (emb_result.data or [])}

            def _cosine(a: Iterable[float], b: Iterable[float]) -> float:
                a_list = list(a)
                b_list = list(b)
                if not a_list or not b_list or len(a_list) != len(b_list):
                    return 0.0
                dot = sum(x * y for x, y in zip(a_list, b_list))
                na = sum(x * x for x in a_list) ** 0.5
                nb = sum(y * y for y in b_list) ** 0.5
                if na == 0 or nb == 0:
                    return 0.0
                return dot / (na * nb)

            scored: list[tuple[float, dict[str, Any]]] = []
            for row in items:
                mid = row.get("id")
                emb = emb_map.get(mid)
                if not emb:
                    continue
                sim = _cosine(q_vec, emb)
                scored.append((sim, row))
            scored.sort(key=lambda t: t[0], reverse=True)
            out: list[dict[str, Any]] = []
            for sim, row in scored[:top_k]:
                row_out = dict(row)
                row_out["similarity"] = float(sim)
                out.append(row_out)
            return out
        except Exception:
            return []


def _detect_preferences_from_exchange(user_message: str) -> tuple[str | None, int | None, list[str]]:
    """Detect preference hints from user message.

    Returns (preferred_language, strictness_delta, topics_add).
    Used by update_profile_from_exchange and maybe_write_episodic_from_exchange.
    """
    text = (user_message or "").strip()
    if not text:
        return None, None, []
    lowered = text.lower()
    preferred_language: str | None = None
    strictness_delta: int | None = None
    topics_add: list[str] = []

    if "arabic only" in lowered or "answer in arabic" in lowered or _has_arabic_script(text):
        preferred_language = "ar"
    elif "english only" in lowered or "answer in english" in lowered:
        preferred_language = "en"

    if "be strict" in lowered or "only answer if" in lowered:
        strictness_delta = 1
    if "be flexible" in lowered or "it's ok if you guess" in lowered:
        strictness_delta = -1 if strictness_delta is None else strictness_delta - 1

    for kw in ("license", "licensing", "capital", "aml", "kyc", "remuneration"):
        if kw in lowered:
            topics_add.append(kw)

    return preferred_language, strictness_delta, topics_add


def update_profile_from_exchange(
    user_id: str,
    session_id: str | None,
    user_message: str,
    assistant_message: str,
) -> None:
    """Heuristic update of user_profile based on a single exchange.

    This is intentionally conservative: it only records clear preferences such
    as language or strictness hints and should never encode regulatory facts.
    """
    if not user_id:
        return
    preferred_language, strictness_delta, topics_add = _detect_preferences_from_exchange(user_message)
    if not any([preferred_language, strictness_delta, topics_add]):
        return

    profile = get_user_profile(user_id) or {}
    new_fields: dict[str, Any] = {}
    if preferred_language:
        new_fields["preferred_language"] = preferred_language
    if strictness_delta is not None:
        current = int(profile.get("strictness_level") or 0)
        updated = max(1, min(5, current + strictness_delta))
        new_fields["strictness_level"] = updated
    if topics_add:
        existing_topics = profile.get("topics") or []
        merged = list(existing_topics)
        for t in topics_add:
            if t not in merged:
                merged.append(t)
        new_fields["topics"] = merged

    if new_fields:
        upsert_user_profile(user_id, **new_fields)


def maybe_write_episodic_from_exchange(
    user_id: str,
    session_id: str | None,
    user_message: str,
    assistant_message: str,
    source_message_id: str | None = None,
) -> None:
    """When preferences are detected, write one episodic memory item and embed it.

    Best-effort; failures are swallowed. Only runs when ENABLE_MEMORY_SYSTEM
    and ENABLE_EPISODIC_MEMORY_WRITES are True.
    """
    if not ENABLE_MEMORY_SYSTEM or not ENABLE_EPISODIC_MEMORY_WRITES or not user_id:
        return
    try:
        preferred_language, strictness_delta, topics_add = _detect_preferences_from_exchange(user_message)
        if not any([preferred_language, strictness_delta, topics_add]):
            return

        parts: list[str] = []
        if preferred_language:
            lang_label = "Arabic" if preferred_language == "ar" else "English"
            parts.append(f"User prefers {lang_label} answers.")
        if strictness_delta is not None:
            direction = "stricter" if strictness_delta > 0 else "more flexible"
            parts.append(f"User wants {direction} answers.")
        if topics_add:
            parts.append(f"User interested in: {', '.join(topics_add)}.")

        text = " ".join(parts).strip()
        if len(text) < MEMORY_ITEM_MIN_LENGTH:
            return

        mid = insert_memory_item(
            user_id=user_id,
            session_id=session_id,
            type_="preference",
            text=text,
            source_message_id=source_message_id,
        )
        if mid:
            embed_memory_item(mid, text)
    except Exception:
        pass


def maybe_update_session_summary(session_id: str, user_id: str) -> None:
    """Update or create a session_summary when enough messages have accumulated.

    Uses the last N exchanges as input to a summarization model. Failures are
    swallowed; the core chat experience must never depend on this succeeding.
    """
    if not session_id or not user_id:
        return
    try:
        # Count messages in this session
        client = get_client()
        count_result = (
            client.table("session_messages")
            .select("message_id", count="exact")
            .eq("session_id", session_id)
            .execute()
        )
        total = int(count_result.count or 0)
        if total <= 0:
            return
        if total % SESSION_SUMMARY_UPDATE_EVERY_N_MESSAGES != 0:
            return

        # Get recent exchanges
        history = get_session_message_history(session_id, limit=20)
        if not history:
            return
        # Build a simple textual transcript
        parts: list[str] = []
        for row in history:
            u = (row.get("user_message") or "").strip()
            a = (row.get("assistant_message") or "").strip()
            if u:
                parts.append(f"User: {u}")
            if a:
                parts.append(f"Assistant: {a}")
        transcript = "\n".join(parts)
        if not transcript:
            return

        # Summarize using the same embedding model's LLM is not available here;
        # for now, store a truncated transcript as a placeholder summary.
        summary_text = transcript[: MEMORY_MAX_CHARS]
        summary_json: dict[str, Any] = {}
        upsert_session_summary(
            session_id=session_id,
            user_id=user_id,
            summary_text=summary_text,
            summary_json=summary_json,
            message_count=total,
        )
    except Exception:
        # Fail soft: do not propagate errors to callers.
        return


def _has_arabic_script(text: str) -> bool:
    """Return True if the text contains Arabic script characters."""
    return any("\u0600" <= c <= "\u06FF" for c in (text or ""))


"""User, session, session_messages, and session_feedback helpers using Supabase."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from supabase_client import get_client


def create_user() -> str:
    """Insert a new row into the user table. Returns user_id (UUID string)."""
    client = get_client()
    result = client.table("user").insert({}).execute()
    if not result.data or len(result.data) == 0:
        raise RuntimeError("create_user: no data returned")
    row = result.data[0]
    user_id = row.get("user_id")
    return str(user_id) if user_id else str(UUID(row["user_id"]))


def create_session(user_id: str) -> str:
    """Insert a new session for the given user_id. Returns session_id (UUID string)."""
    client = get_client()
    result = client.table("session").insert({"user_id": user_id}).execute()
    if not result.data or len(result.data) == 0:
        raise RuntimeError("create_session: no data returned")
    row = result.data[0]
    session_id = row.get("session_id")
    return str(session_id) if session_id else str(UUID(row["session_id"]))


def insert_session_message(
    session_id: str,
    user_id: str,
    user_message: str,
    assistant_message: str,
    timestamp: datetime | None = None,
) -> str:
    """Insert a message row. Returns message_id (UUID string)."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    client = get_client()
    data: dict[str, Any] = {
        "session_id": session_id,
        "user_id": user_id,
        "user_message": user_message,
        "assistant_message": assistant_message,
        "timestamp": timestamp.isoformat(),
    }
    result = client.table("session_messages").insert(data).execute()
    if not result.data or len(result.data) == 0:
        raise RuntimeError("insert_session_message: no data returned")
    row = result.data[0]
    message_id = row.get("message_id")
    return str(message_id) if message_id else str(UUID(row["message_id"]))


def get_session_message_history(
    session_id: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return the last `limit` message exchanges for the session, in chronological order (oldest first)."""
    client = get_client()
    result = (
        client.table("session_messages")
        .select("user_message, assistant_message, timestamp")
        .eq("session_id", session_id)
        .order("timestamp", desc=True)
        .limit(limit)
        .execute()
    )
    rows = list(result.data or [])
    rows.reverse()
    return rows


def upsert_session_feedback(
    session_id: str,
    user_id: str,
    message_id: str,
    feedback: int,
    comments: str | None = None,
    timestamp: datetime | None = None,
) -> None:
    """Insert or update feedback for a message. feedback must be 0 or 1."""
    if feedback not in (0, 1):
        raise ValueError("feedback must be 0 or 1")
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    client = get_client()
    data: dict[str, Any] = {
        "session_id": session_id,
        "user_id": user_id,
        "message_id": message_id,
        "feedback": feedback,
        "timestamp": timestamp.isoformat(),
    }
    if comments is not None:
        data["comments"] = comments
    client.table("session_feedback").upsert(
        data,
        on_conflict="session_id,message_id",
    ).execute()

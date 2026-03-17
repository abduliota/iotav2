"""
IOTAV3 users/sessions wrapper.

Re-exports the small helper surface from the main `backend/users_sessions.py`
so that the IOTAV3 backend can import from a local namespace.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, List, Dict

from users_sessions import (  # type: ignore
    create_session as _create_session,
    create_user as _create_user,
    get_message_by_id as _get_message_by_id,
    get_session_message_history as _get_session_message_history,
    insert_session_message as _insert_session_message,
    upsert_session_feedback as _upsert_session_feedback,
)

__all__ = [
    "create_user",
    "create_session",
    "insert_session_message",
    "get_session_message_history",
    "get_message_by_id",
    "upsert_session_feedback",
]


def create_user() -> str:
    return _create_user()


def create_session(user_id: str) -> str:
    return _create_session(user_id)


def insert_session_message(
    session_id: str,
    user_id: str,
    user_message: str,
    assistant_message: str,
    timestamp: datetime | None = None,
) -> str:
    return _insert_session_message(
        session_id=session_id,
        user_id=user_id,
        user_message=user_message,
        assistant_message=assistant_message,
        timestamp=timestamp,
    )


def get_session_message_history(
    session_id: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    return _get_session_message_history(session_id, limit=limit)


def get_message_by_id(message_id: str) -> dict[str, Any] | None:
    return _get_message_by_id(message_id)


def upsert_session_feedback(
    session_id: str,
    user_id: str,
    message_id: str,
    feedback: int,
    comments: str | None = None,
    user_message: str | None = None,
    assistant_message: str | None = None,
    timestamp: datetime | None = None,
) -> None:
    _upsert_session_feedback(
      session_id=session_id,
      user_id=user_id,
      message_id=message_id,
      feedback=feedback,
      comments=comments,
      user_message=user_message,
      assistant_message=assistant_message,
      timestamp=timestamp,
    )


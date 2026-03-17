"""
IOTAV3 Supabase client wrapper.

Delegates to the existing `backend/supabase_client.py` implementation so
that IOTAV3 code can import from `IOTAV3.backend.db.supabase_client`.
"""

from __future__ import annotations

from typing import Any

from supabase_client import get_client as _get_client  # type: ignore

__all__ = ["get_client"]


def get_client():
    return _get_client()


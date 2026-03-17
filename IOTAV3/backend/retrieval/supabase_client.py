from __future__ import annotations

from typing import Any

import os
from supabase import create_client, Client  # type: ignore


_client: Client | None = None


def get_client() -> Client:
    """
    Return a singleton Supabase client for IOTAV3.

    Configuration is read from the IOTAV3 backend .env:
    - SUPABASE_URL
    - SUPABASE_SERVICE_ROLE_KEY
    """

    global _client
    if _client is not None:
        return _client

    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    _client = create_client(url, key)
    return _client


def rpc(function: str, params: dict[str, Any]) -> Any:
    """
    Convenience wrapper to call a Supabase RPC.
    """

    client = get_client()
    return client.rpc(function, params).execute()


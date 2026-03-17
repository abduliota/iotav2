from __future__ import annotations

from typing import Any, Dict, List

from IOTAV3.backend import config_iotav3 as cfg
from IOTAV3.backend.retrieval.supabase_client import rpc


def retrieve_chunks(query_embedding: List[float], top_k: int | None = None) -> List[Dict[str, Any]]:
    """
    Fetch top‑k chunks by vector similarity via Supabase `match_chunks` RPC.

    This mirrors the legacy behaviour but keeps a much smaller surface
    tailored to the IOTAV3 pipeline.
    """

    if top_k is None:
        top_k = cfg.TOP_K

    result = rpc(
        "match_chunks",
        {
            "query_embedding": query_embedding,
            "match_count": top_k,
            "snippet_char_limit": cfg.SNIPPET_CHAR_LIMIT,
        },
    )
    return result.data or []


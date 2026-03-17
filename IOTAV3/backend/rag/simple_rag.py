"""
IOTAV3 RAG wrapper.

For this iteration we delegate to the existing `simple_rag` module in
the root backend. This keeps behavior identical while allowing the
IOTAV3 backend to import a stable interface from its own namespace.

If we later decide to fork and simplify the RAG pipeline, this module
is the right place to do so.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from simple_rag import (  # type: ignore
    answer_query as _answer_query,
    generate_answer_simple as _generate_answer_simple,
)

AnswerResult = dict[str, Any]


def answer_query(
    user_query: str,
    top_k: int | None = None,
    category: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    on_chunk: Optional[Callable[[str], None]] = None,
) -> AnswerResult:
    """Thin wrapper around the existing `simple_rag.answer_query`."""

    return _answer_query(
        user_query,
        top_k=top_k,
        category=category,
        user_id=user_id,
        session_id=session_id,
        on_chunk=on_chunk,
    )


def generate_answer_simple(*args: Any, **kwargs: Any) -> str:
    """Re-export `generate_answer_simple` for completeness."""

    return _generate_answer_simple(*args, **kwargs)


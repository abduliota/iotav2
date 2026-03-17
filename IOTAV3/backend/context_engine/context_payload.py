from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Passage:
    """
    Single retrieved passage used as part of the RAG context.

    The pipeline keeps this intentionally small: document identifier,
    page range, short content snippet, and similarity score used only
    for ordering / diagnostics.
    """

    document_name: str
    page_start: int
    page_end: int
    snippet: str
    similarity: float


@dataclass
class ContextPayload:
    """
    Structured view of all context fed to the LLM.

    This is later flattened into text in a consistent way by the prompt
    assembly logic, but keeping it structured makes it easier to test
    and to plug in richer UIs later.
    """

    passages: List[Passage]
    session_summary: Optional[str]
    user_profile_summary: Optional[str]
    cag_block: str


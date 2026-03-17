"""
Standalone CAG (Cached Augmented Generation) layer for IOTAV3.

This keeps a small amount of stable, curated background knowledge and
rules that should accompany most RAG queries, without requiring another
round‑trip to the vector database.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class CachedKnowledge:
    """
    Structured representation of cached knowledge.

    For now this is a thin wrapper around a couple of text blocks, but it
    can be extended later with richer structure (FAQ entries, domain
    glossary, etc.) without touching callers.
    """

    system_rules: str
    domain_background: str

    def as_text_block(self) -> str:
        """Return a single text block suitable for inclusion in prompts."""
        parts: list[str] = []
        if self.system_rules.strip():
            parts.append("### SYSTEM RULES\n\n" + self.system_rules.strip())
        if self.domain_background.strip():
            parts.append(
                "### DOMAIN BACKGROUND\n\n" + self.domain_background.strip()
            )
        return "\n\n".join(parts).strip()


_DEFAULT_CACHED_KNOWLEDGE = CachedKnowledge(
    system_rules=(
        "You are a Saudi GRC compliance assistant. You MUST answer only "
        "using the provided regulatory context. If the context does not "
        "support a factual answer, you must respond with the configured "
        "not-found message.\n\n"
        "User-specific memory (USER CONTEXT) is never regulatory evidence. "
        "It may be used only to adjust tone, language, or focus. If USER "
        "CONTEXT ever conflicts with the regulatory context, you MUST ignore "
        "USER CONTEXT and follow the official documents."
    ),
    domain_background=(
        "You specialise in Saudi governance, risk and compliance frameworks, "
        "including SAMA and NORA banking regulations, the Aramco "
        "Cybersecurity Compliance Certificate (CCC) program, NCA Essential "
        "Cybersecurity Controls (ECC), the Saudi Personal Data Protection "
        "Law (PDPL), and ISO 27k information security standards.\n\n"
        "Typical questions involve licensing, risk management, information "
        "security controls, privacy rights and obligations, and audit or "
        "compliance evidence. Answers should stay strictly within these "
        "frameworks and prefer verbatim or near-verbatim text from the "
        "retrieved context, with page-based citations where available."
    ),
)


def load_cag(_: Dict[str, Any] | None = None) -> CachedKnowledge:
    """
    Return the current cached knowledge object.

    The optional context argument is reserved for future per-tenant/domain
    selection; for now it is ignored.
    """

    return _DEFAULT_CACHED_KNOWLEDGE


def get_cag_text_block(context: Dict[str, Any] | None = None) -> str:
    """
    Convenience helper: return the cached knowledge as a single text block.
    """

    return load_cag(context).as_text_block()


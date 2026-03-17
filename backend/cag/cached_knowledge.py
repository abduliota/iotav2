"""CAG (Cached Augmented Generation) static knowledge layer.

This module centralizes stable, domain-specific instructions and background
knowledge that should be available to the RAG pipeline on every request.

The goal is to:
- Keep core prompts and domain rules in one place.
- Make it easy to version / extend cached knowledge later (per-tenant,
  per-domain, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class CachedKnowledge:
    """Structured representation of cached knowledge.

    For now this is a thin wrapper around a couple of text blocks, but it can
    be extended later with richer structure (FAQ entries, domain glossary,
    etc.) without touching callers.
    """

    system_rules: str
    domain_background: str

    def as_text_block(self) -> str:
        """Return a single text block suitable for inclusion in prompts."""
        parts: list[str] = []
        if self.system_rules.strip():
            parts.append("### SYSTEM RULES\n\n" + self.system_rules.strip())
        if self.domain_background.strip():
            parts.append("### DOMAIN BACKGROUND\n\n" + self.domain_background.strip())
        return "\n\n".join(parts).strip()


# Default, static cached knowledge. This intentionally avoids referencing any
# runtime configuration so it can be imported cheaply from hot paths.
_DEFAULT_CACHED_KNOWLEDGE = CachedKnowledge(
    system_rules=(
        "You are a compliance assistant specialized in SAMA/NORA Saudi banking "
        "regulations. You MUST answer only using the provided regulatory "
        "context. If the context does not support a factual answer, you must "
        "respond with the configured not-found message.\n\n"
        "User-specific memory (USER CONTEXT) is never regulatory evidence. It "
        "may be used only to adjust tone, language, or focus. If USER CONTEXT "
        "ever conflicts with the regulatory context, you MUST ignore USER "
        "CONTEXT and follow the regulatory documents."
    ),
    domain_background=(
        "SAMA/NORA documents include laws, rulebooks, circulars, and "
        "guidelines that govern the Saudi banking sector. Typical user "
        "questions involve licensing requirements, capital adequacy, risk "
        "management, governance, AML/KYC, outsourcing, remuneration, and "
        "related-party exposures.\n\n"
        "Answers should:\n"
        "- Stay strictly within the scope of SAMA/NORA documents.\n"
        "- Prefer verbatim or near-verbatim text from the retrieved context.\n"
        "- Include page-based citations where available.\n"
        "- Avoid generic banking advice that is not grounded in the context."
    ),
)


def load_cag(_: Dict[str, Any] | None = None) -> CachedKnowledge:
    """Return the current cached knowledge object.

    The optional context argument is reserved for future per-tenant/domain
    selection; for now it is ignored.
    """

    return _DEFAULT_CACHED_KNOWLEDGE


def get_cag_text_block(context: Dict[str, Any] | None = None) -> str:
    """Convenience helper: return the cached knowledge as a single text block."""

    return load_cag(context).as_text_block()


"""Simple regulatory ontology: Law -> Regulation -> Circular for metadata filter or prompt."""
from __future__ import annotations

# Hierarchy: doc_type -> parent (None for root)
DOC_TYPE_HIERARCHY: dict[str, str | None] = {
    "Law": None,
    "Regulation": "Law",
    "Circular": "Regulation",
    "Guideline": "Regulation",
}

# Aliases for filtering
DOC_TYPE_ALIASES: dict[str, str] = {
    "law": "Law",
    "regulation": "Regulation",
    "circular": "Circular",
    "guideline": "Guideline",
    "قانون": "Law",
    "لائحة": "Regulation",
    "تعميم": "Circular",
}


def normalize_doc_type(doc_type: str | None) -> str | None:
    """Return canonical doc_type (Law, Regulation, Circular, Guideline) or None."""
    if not doc_type or not doc_type.strip():
        return None
    key = doc_type.strip()
    return DOC_TYPE_ALIASES.get(key.lower(), key) if key else None


def get_parent_doc_type(doc_type: str) -> str | None:
    """Return parent in hierarchy (e.g. Regulation -> Law)."""
    canonical = normalize_doc_type(doc_type)
    return DOC_TYPE_HIERARCHY.get(canonical or "", None)

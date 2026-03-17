"""
IOTAV3 Qwen loader wrapper.

Exposes `_load_qwen` from the main backend `qwen_model` module so the
IOTAV3 RAG engine can import from a local namespace.
"""

from __future__ import annotations

from typing import Tuple

from qwen_model import _load_qwen as _core_load_qwen  # type: ignore

__all__ = ["_load_qwen"]


def _load_qwen():
    return _core_load_qwen()


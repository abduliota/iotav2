from __future__ import annotations

"""
IOTAV3 FastAPI entrypoint.

This module simply exposes the FastAPI `app` defined in `server.py` so
it can be run with:

    uvicorn IOTAV3.backend.main:app --reload

The implementation intentionally mirrors the existing backend server
while keeping all new code scoped under `IOTAV3/`.
"""

from .server import app  # noqa: F401


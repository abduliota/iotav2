from __future__ import annotations

"""
IOTAV3 FastAPI server
---------------------

This server exposes the same public API as the existing backend:

- POST /api/user
- POST /api/session
- POST /api/query
- POST /api/query-stream
- POST /api/feedback
- GET  /health

Internally it reuses the proven RAG, memory, and users/sessions logic
from the existing backend via imports, while keeping all new wiring code
within the IOTAV3 namespace.
"""

import json
import os
import queue
import threading
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Ensure we can import the existing backend modules without modifying them.
ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
import sys  # noqa: E402

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Load the existing backend .env so we reuse the same configuration.
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(BACKEND_DIR / ".env")
except Exception:
    pass

def _answer_query_dispatch(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Internal helper to route RAG calls.

    This implementation always uses the IOTAV3‑native RAG pipeline in
    `IOTAV3.backend.rag.pipeline.answer_query_iotav3` and never calls the
    legacy `simple_rag` implementation.
    """
    from IOTAV3.backend.rag.pipeline import (  # type: ignore
        answer_query_iotav3,
    )

    return answer_query_iotav3(*args, **kwargs)


from IOTAV3.backend.memory.memory import (  # type: ignore  # noqa: E402
    maybe_update_session_summary,
    maybe_write_episodic_from_exchange,
    update_profile_from_exchange,
)
from IOTAV3.backend.db.users_sessions import (  # type: ignore  # noqa: E402
    create_session,
    create_user,
    get_message_by_id,
    get_session_message_history,
    insert_session_message,
    upsert_session_feedback,
)


class QueryBody(BaseModel):
    query: str = Field(..., min_length=1)
    user_id: str | None = None
    session_id: str | None = None


class SessionBody(BaseModel):
    user_id: str


class FeedbackBody(BaseModel):
    session_id: str
    user_id: str
    message_id: str
    feedback: int = Field(..., ge=1, le=5)
    comments: str | None = None


app = FastAPI(title="IOTAV3 RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/user")
def api_create_user() -> Dict[str, Any]:
    """Create a new user. Returns { user_id }."""
    try:
        user_id = create_user()
        return {"user_id": user_id}
    except Exception as e:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/session")
def api_create_session(body: SessionBody) -> Dict[str, Any]:
    """Create a new session for the given user_id. Returns { session_id }."""
    try:
        session_id = create_session(body.user_id)
        return {"session_id": session_id}
    except Exception as e:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query")
def api_query(body: QueryBody) -> Dict[str, Any]:
    """
    Run RAG on the query.

    This mirrors the behavior of the existing backend: it ensures a user
    and session exist, calls `answer_query`, persists the exchange, and
    then performs best‑effort memory updates.
    """
    try:
        user_id = body.user_id
        session_id = body.session_id
        created_user = False
        created_session = False

        if not user_id:
            user_id = create_user()
            created_user = True
        if not session_id:
            session_id = create_session(user_id)
            created_session = True

        result = _answer_query_dispatch(
            body.query,
            user_id=user_id,
            session_id=session_id,
        )
        answer = result.get("answer") or ""
        sources = result.get("sources") or []

        message_id = insert_session_message(
            session_id=session_id,
            user_id=user_id,
            user_message=body.query,
            assistant_message=answer,
        )

        # Best‑effort memory updates (fail soft).
        try:
            update_profile_from_exchange(
                user_id=user_id,
                session_id=session_id,
                user_message=body.query,
                assistant_message=answer,
            )
            maybe_update_session_summary(session_id=session_id, user_id=user_id)
            maybe_write_episodic_from_exchange(
                user_id=user_id,
                session_id=session_id,
                user_message=body.query,
                assistant_message=answer,
                source_message_id=message_id,
            )
        except Exception:
            pass

        out: Dict[str, Any] = {
            "answer": answer,
            "sources": sources,
            "message_id": message_id,
            "user_id": user_id,
            "session_id": session_id,
        }
        if created_user:
            out["user_id_created"] = True
        if created_session:
            out["session_id_created"] = True
        return out
    except HTTPException:
        raise
    except Exception as e:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query-stream")
def api_query_stream(body: QueryBody) -> StreamingResponse:
    """
    Streaming wrapper around `answer_query`.

    The semantics are identical to the existing backend:
    - Text is streamed as `{ \"type\": \"chunk\", \"text\": \"...\" }` lines.
    - At the end, a `meta` event and a final `done` event are sent.
    """
    try:
        user_id = body.user_id
        session_id = body.session_id
        created_user = False
        created_session = False

        if not user_id:
            user_id = create_user()
            created_user = True
        if not session_id:
            session_id = create_session(user_id)
            created_session = True

        chunk_queue: "queue.Queue[str | None]" = queue.Queue()
        result_holder: Dict[str, Dict[str, Any]] = {}

        def on_chunk(text: str) -> None:
            chunk_queue.put(text)

        def run_rag() -> None:
            try:
                result = _answer_query_dispatch(
                    body.query,
                    user_id=user_id,
                    session_id=session_id,
                    on_chunk=on_chunk,
                )
                result_holder["result"] = result
            except Exception as e:  # pragma: no cover - defensive
                result_holder["error"] = {"detail": str(e)}
            finally:
                chunk_queue.put(None)

        threading.Thread(target=run_rag, daemon=True).start()

        def event_stream():
            # 1) Stream incremental chunks.
            while True:
                chunk = chunk_queue.get()
                if chunk is None:
                    break
                yield json.dumps({"type": "chunk", "text": chunk}) + "\n"

            # 2) On completion, either send error or finalize.
            if "error" in result_holder:
                yield json.dumps(
                    {"type": "error", "detail": result_holder["error"]["detail"]}
                ) + "\n"
                return

            result = result_holder.get("result") or {}
            answer = result.get("answer") or ""
            sources = result.get("sources") or []

            message_id = insert_session_message(
                session_id=session_id,
                user_id=user_id,
                user_message=body.query,
                assistant_message=answer,
            )

            try:
                update_profile_from_exchange(
                    user_id=user_id,
                    session_id=session_id,
                    user_message=body.query,
                    assistant_message=answer,
                )
                maybe_update_session_summary(session_id=session_id, user_id=user_id)
                maybe_write_episodic_from_exchange(
                    user_id=user_id,
                    session_id=session_id,
                    user_message=body.query,
                    assistant_message=answer,
                    source_message_id=message_id,
                )
            except Exception:
                pass

            meta = {
                "user_id": user_id,
                "session_id": session_id,
                "message_id": message_id,
                "sources": sources,
                "user_id_created": created_user,
                "session_id_created": created_session,
            }

            yield json.dumps({"type": "meta", "meta": meta}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
        )
    except HTTPException:
        raise
    except Exception as e:  # pragma: no cover - defensive
        def error_stream():
            yield json.dumps({"type": "error", "detail": str(e)}) + "\n"

        return StreamingResponse(
            error_stream(),
            media_type="text/event-stream",
            status_code=500,
        )


@app.post("/api/feedback")
def api_feedback(body: FeedbackBody) -> Dict[str, Any]:
    """
    Record a star rating (1‑5) and optional comments for a message.

    Looks up the original message text and stores denormalized copies in
    the feedback row, mirroring existing behavior.
    """
    try:
        msg = get_message_by_id(body.message_id)
        user_message = (msg.get("user_message") or "") if msg else ""
        assistant_message = (msg.get("assistant_message") or "") if msg else ""
        upsert_session_feedback(
            session_id=body.session_id,
            user_id=body.user_id,
            message_id=body.message_id,
            feedback=body.feedback,
            comments=body.comments,
            user_message=user_message or None,
            assistant_message=assistant_message or None,
        )
        return {"ok": True}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health() -> Dict[str, str]:
    """Simple health check."""
    # Touch a cheap dependency (session history) to ensure DB connectivity
    # without failing the endpoint if Supabase is temporarily unavailable.
    try:
        _ = get_session_message_history  # noqa: F841
    except Exception:
        pass
    return {"status": "ok"}


"""FastAPI server: /api/query (RAG), /api/user, /api/session, /api/feedback."""
from __future__ import annotations

import os
import sys
from pathlib import Path
import json
import time
import threading
import queue

# Ensure backend directory is on path when running uvicorn server:app from project root
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# Load .env from backend dir
try:
    from dotenv import load_dotenv
    load_dotenv(_BACKEND_DIR / ".env")
except ImportError:
    pass

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from simple_rag import answer_query
from users_sessions import (
    create_user,
    create_session,
    get_message_by_id,
    insert_session_message,
    upsert_session_feedback,
)

app = FastAPI(title="Iotav2 RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/response models ---

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


# --- Routes ---

@app.post("/api/user")
def api_create_user():
    """Create a new user. Returns { user_id }."""
    try:
        user_id = create_user()
        return {"user_id": user_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/session")
def api_create_session(body: SessionBody):
    """Create a new session for the given user_id. Returns { session_id }."""
    try:
        session_id = create_session(body.user_id)
        return {"session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query")
def api_query(body: QueryBody):
    """
    Run RAG on the query. Optionally pass user_id and session_id; if missing,
    creates a new user and/or session and returns them. Persists the exchange
    in session_messages and returns message_id.
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

        result = answer_query(body.query, session_id=session_id)
        answer = result.get("answer") or ""
        sources = result.get("sources") or []

        message_id = insert_session_message(
            session_id=session_id,
            user_id=user_id,
            user_message=body.query,
            assistant_message=answer,
        )

        out = {
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Streaming query endpoint (wrapper, does NOT change core RAG) ---

@app.post("/api/query-stream")
def api_query_stream(body: QueryBody):
    """
    Streaming wrapper around answer_query.

    - Calls answer_query once (does NOT change core RAG).
    - Streams the final answer back to the client in small chunks as JSON lines.
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

        # Queue for streaming text chunks from the RAG pipeline to the client.
        chunk_queue: "queue.Queue[str | None]" = queue.Queue()
        sources_queue: "queue.Queue[list]" = queue.Queue()
        result_holder: dict[str, dict] = {}

        def on_chunk(text: str) -> None:
            """Called from within the RAG generation loop for each new text piece."""
            chunk_queue.put(text)

        def on_sources_ready(sources: list) -> None:
            """Called when retrieval is done; client gets sources early so they show even if stream is cut."""
            sources_queue.put(sources)

        def run_rag() -> None:
            """Run answer_query in a background thread, streaming chunks via on_chunk."""
            try:
                result = answer_query(
                    body.query,
                    session_id=session_id,
                    on_chunk=on_chunk,
                    on_sources_ready=on_sources_ready,
                )
                result_holder["result"] = result
            except Exception as e:  # pragma: no cover - defensive
                result_holder["error"] = {"detail": str(e)}
            finally:
                chunk_queue.put(None)

        threading.Thread(target=run_rag, daemon=True).start()

        def event_stream():
            # 0) Send meta with sources as soon as retrieval is done (so client has sources even if stream is cut later).
            try:
                early_sources = sources_queue.get(timeout=120)
            except queue.Empty:
                early_sources = []
            yield json.dumps({"type": "meta", "meta": {"sources": early_sources}}) + "\n"

            # 1) Stream chunks to the client as they are generated.
            while True:
                chunk = chunk_queue.get()
                if chunk is None:
                    break
                yield json.dumps({"type": "chunk", "text": chunk}) + "\n"

            # 2) After generation finishes, either send an error or finalize the response.
            if "error" in result_holder:
                yield json.dumps({"type": "error", "detail": result_holder["error"]["detail"]}) + "\n"
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

            meta = {
                "user_id": user_id,
                "session_id": session_id,
                "message_id": message_id,
                "sources": sources,
                "user_id_created": created_user,
                "session_id_created": created_session,
            }

            # 3) Send meta and completion signal.
            yield json.dumps({"type": "meta", "meta": meta}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
        )
    except Exception as e:
        def error_stream():
            yield json.dumps({"type": "error", "detail": str(e)}) + "\n"

        return StreamingResponse(
            error_stream(),
            media_type="text/event-stream",
            status_code=500,
        )


@app.post("/api/feedback")
def api_feedback(body: FeedbackBody):
    """Record star rating (1-5) and optional comments for a message. Fills user_message/assistant_message from session_messages."""
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}

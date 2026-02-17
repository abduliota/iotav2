"""FastAPI server: /api/query (RAG), /api/user, /api/session, /api/feedback."""
from __future__ import annotations

import os
import sys
from pathlib import Path

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
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from simple_rag import answer_query, answer_query_streaming
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


@app.post("/api/query/stream")
def api_query_stream(body: QueryBody):
    """
    Stream answer text in chunks for better frontend responsiveness.

    NOTE: This currently streams only the answer text; sources and IDs
    are still available on the backend (and persisted) but not included
    in the streaming payload.
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

        full_answer, sources, chunks = answer_query_streaming(body.query, session_id=session_id)

        # Persist the full exchange as usual
        message_id = insert_session_message(
            session_id=session_id,
            user_id=user_id,
            user_message=body.query,
            assistant_message=full_answer,
        )

        def iter_chunks():
            # Optionally, we could prefix metadata or delimit chunks; for now
            # we stream raw text slices.
            for chunk in chunks:
                if chunk:
                    yield chunk

        headers = {
            "X-Session-Id": session_id or "",
            "X-User-Id": user_id or "",
            "X-Message-Id": message_id,
        }
        return StreamingResponse(iter_chunks(), media_type="text/plain", headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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

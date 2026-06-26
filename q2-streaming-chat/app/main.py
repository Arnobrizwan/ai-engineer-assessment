"""FastAPI application: streaming chat API + static single-page UI.

Routes
------
``POST   /api/chat``          Stream the assistant reply token-by-token (SSE).
``GET    /api/history``       Return the persisted message history for a session.
``POST   /api/session/reset`` Clear all messages for a session.
``DELETE /api/session``       Alias for reset (RESTful delete).
``GET    /healthz``           Liveness probe.
``GET    /``                  Serves the chat UI (``static/index.html``).

SSE streaming sequence for ``POST /api/chat``::

    browser --POST /api/chat--> FastAPI
        FastAPI: load prior history from SQLite
        FastAPI: persist the new user message
        FastAPI --stream--> Ollama (/api/chat, stream=true)
        Ollama  --tokens--> FastAPI --"data: <token>"--> browser  (live)
        FastAPI: persist the full assistant reply on completion
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Dict, List

from fastapi import Depends, FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from . import db as db_module
from .config import get_settings
from .llm import ChatMessage, stream_chat
from .schemas import (
    ChatRequest,
    HistoryResponse,
    MessageOut,
    ResetRequest,
    ResetResponse,
)

SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. Use Markdown for formatting when "
    "it improves clarity."
)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Initialize the database schema on startup."""

    db_module.init_db()
    yield


app = FastAPI(title="Streaming LLM Chat", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------


def build_context(history: List[db_module.Message]) -> List[ChatMessage]:
    """Convert persisted messages into Ollama chat-format context.

    A system prompt is prepended so the model always has guidance, followed by
    every prior turn in chronological order. This is what gives the assistant
    memory of earlier messages in the session.

    Args:
        history: Ordered list of persisted :class:`~app.db.Message` rows
            (already including the latest user message).

    Returns:
        A list of ``{"role", "content"}`` dicts ready to send to Ollama.
    """

    context: List[ChatMessage] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for message in history:
        context.append({"role": message.role, "content": message.content})
    return context


def _sse(data: Dict[str, str]) -> str:
    """Format a payload as a single Server-Sent Events ``data:`` frame.

    Args:
        data: JSON-serializable payload.

    Returns:
        An SSE frame terminated by a blank line.
    """

    return f"data: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# Streaming chat endpoint
# ---------------------------------------------------------------------------


@app.post("/api/chat")
async def chat(payload: ChatRequest, db: Session = Depends(db_module.get_db)) -> StreamingResponse:
    """Stream the assistant's reply token-by-token over SSE.

    The handler persists the incoming user message, assembles the full
    conversation context, streams tokens from Ollama to the client as they
    arrive, and finally persists the complete assistant reply so it becomes
    part of the session memory.

    Args:
        payload: Validated chat request (message + session id).
        db: Injected database session.

    Returns:
        A ``text/event-stream`` :class:`StreamingResponse`.
    """

    session_id = payload.session_id

    # Persist the user message first, then load the full ordered history so the
    # latest turn is included in the context sent to the model.
    db_module.add_message(db, session_id, "user", payload.message)
    history = db_module.get_history(db, session_id)
    context = build_context(history)

    async def event_generator() -> AsyncIterator[str]:
        """Yield SSE frames: a start marker, tokens, then a done marker."""

        yield _sse({"type": "start", "session_id": session_id})
        parts: List[str] = []
        try:
            async for token in stream_chat(context):
                parts.append(token)
                yield _sse({"type": "token", "content": token})
        except Exception as exc:  # noqa: BLE001 - surface error to the client
            yield _sse({"type": "error", "content": f"LLM error: {exc}"})
        finally:
            full_reply = "".join(parts)
            if full_reply:
                # Persist the assistant reply so future turns remember it.
                db_module.add_message(db, session_id, "assistant", full_reply)
            yield _sse({"type": "done", "content": full_reply})

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # disable proxy buffering for true streaming
    }
    return StreamingResponse(
        event_generator(), media_type="text/event-stream", headers=headers
    )


# ---------------------------------------------------------------------------
# History / reset endpoints
# ---------------------------------------------------------------------------


@app.get("/api/history", response_model=HistoryResponse)
def history(
    session_id: str = "default", db: Session = Depends(db_module.get_db)
) -> HistoryResponse:
    """Return the persisted message history for a session.

    Args:
        session_id: Session identifier (defaults to the shared session).
        db: Injected database session.

    Returns:
        A :class:`HistoryResponse` with messages in chronological order.
    """

    messages = db_module.get_history(db, session_id)
    return HistoryResponse(
        session_id=session_id,
        messages=[MessageOut.model_validate(m) for m in messages],
    )


@app.post("/api/session/reset", response_model=ResetResponse)
def reset_session(
    payload: ResetRequest, db: Session = Depends(db_module.get_db)
) -> ResetResponse:
    """Clear all messages for a session.

    Args:
        payload: Reset request containing the session id.
        db: Injected database session.

    Returns:
        A :class:`ResetResponse` with the number of deleted messages.
    """

    deleted = db_module.reset_session(db, payload.session_id)
    return ResetResponse(session_id=payload.session_id, deleted=deleted)


@app.delete("/api/session", response_model=ResetResponse)
def delete_session(
    session_id: str = "default", db: Session = Depends(db_module.get_db)
) -> ResetResponse:
    """RESTful alias for resetting a session via ``DELETE``.

    Args:
        session_id: Session identifier to clear.
        db: Injected database session.

    Returns:
        A :class:`ResetResponse` with the number of deleted messages.
    """

    deleted = db_module.reset_session(db, session_id)
    return ResetResponse(session_id=session_id, deleted=deleted)


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    """Liveness probe used by Docker/orchestration health checks."""

    return {"status": "ok", "model": get_settings().llm_model}


# ---------------------------------------------------------------------------
# Static UI
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the single-page chat UI."""

    return FileResponse(str(STATIC_DIR / "index.html"))


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

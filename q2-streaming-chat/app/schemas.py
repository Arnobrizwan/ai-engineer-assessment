"""Pydantic schemas for request/response validation."""

from __future__ import annotations

from datetime import datetime
from typing import List, Literal

from pydantic import BaseModel, Field

DEFAULT_SESSION_ID = "default"

Role = Literal["user", "assistant", "system"]


class ChatRequest(BaseModel):
    """Body for ``POST /api/chat``.

    Attributes:
        message: The user's new message. Must be non-empty.
        session_id: Conversation identifier; defaults to a single shared
            session so the API is usable with zero client state.
    """

    message: str = Field(..., min_length=1, description="User message text.")
    session_id: str = Field(
        default=DEFAULT_SESSION_ID,
        min_length=1,
        max_length=64,
        description="Conversation/session identifier.",
    )


class MessageOut(BaseModel):
    """Serialized message returned by history/reset endpoints."""

    id: int
    role: Role
    content: str
    created_at: datetime

    model_config = {"from_attributes": True}


class HistoryResponse(BaseModel):
    """Response body for ``GET /api/history``."""

    session_id: str
    messages: List[MessageOut]


class ResetRequest(BaseModel):
    """Body for ``POST /api/session/reset``."""

    session_id: str = Field(default=DEFAULT_SESSION_ID, min_length=1, max_length=64)


class ResetResponse(BaseModel):
    """Response body for a successful session reset."""

    session_id: str
    deleted: int

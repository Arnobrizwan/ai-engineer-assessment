"""Database layer: SQLAlchemy models and persistence helpers.

The schema is intentionally small and consists of two tables:

``sessions``
    One row per chat session. Identified by a client-supplied string id.

``messages``
    One row per message (user or assistant), linked to a session via a foreign
    key. Messages are ordered by ``created_at`` (and ``id`` as a tie-breaker)
    so the conversation history can be reconstructed deterministically.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterator, List

from sqlalchemy import (
    DateTime,
    ForeignKey,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from .config import get_settings


def _utcnow() -> datetime:
    """Return the current timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base class for all ORM models."""


class ChatSession(Base):
    """A single chat conversation.

    Attributes:
        id: Client-supplied unique session identifier.
        created_at: Creation timestamp (UTC).
        messages: Ordered list of messages belonging to this session.
    """

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)

    messages: Mapped[List["Message"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="Message.created_at, Message.id",
    )


class Message(Base):
    """A single message within a chat session.

    Attributes:
        id: Auto-incrementing primary key.
        session_id: Foreign key to the owning session.
        role: One of ``"user"``, ``"assistant"`` or ``"system"``.
        content: The message text.
        created_at: Creation timestamp (UTC), used for ordering.
    """

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), index=True, nullable=False
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow, nullable=False)

    session: Mapped[ChatSession] = relationship(back_populates="messages")


# ---------------------------------------------------------------------------
# Engine / session factory
# ---------------------------------------------------------------------------

_settings = get_settings()

# ``check_same_thread`` is required because Uvicorn serves requests across
# threads while SQLite connections are otherwise pinned to one thread.
_connect_args = (
    {"check_same_thread": False} if _settings.database_url.startswith("sqlite") else {}
)

engine = create_engine(_settings.database_url, connect_args=_connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def init_db() -> None:
    """Create all tables if they do not already exist."""

    Base.metadata.create_all(bind=engine)


def get_db() -> Iterator[Session]:
    """FastAPI dependency that yields a database session and closes it.

    Yields:
        An active SQLAlchemy :class:`~sqlalchemy.orm.Session`.
    """

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Persistence helpers (pure functions operating on a Session)
# ---------------------------------------------------------------------------


def get_or_create_session(db: Session, session_id: str) -> ChatSession:
    """Fetch an existing session or create it if missing.

    Args:
        db: Active database session.
        session_id: Unique session identifier.

    Returns:
        The persisted :class:`ChatSession` instance.
    """

    session = db.get(ChatSession, session_id)
    if session is None:
        session = ChatSession(id=session_id)
        db.add(session)
        db.commit()
        db.refresh(session)
    return session


def add_message(db: Session, session_id: str, role: str, content: str) -> Message:
    """Append a message to a session, creating the session if needed.

    Args:
        db: Active database session.
        session_id: Identifier of the owning session.
        role: Message role (``"user"``/``"assistant"``/``"system"``).
        content: Message body.

    Returns:
        The newly persisted :class:`Message`.
    """

    get_or_create_session(db, session_id)
    message = Message(session_id=session_id, role=role, content=content)
    db.add(message)
    db.commit()
    db.refresh(message)
    return message


def get_history(db: Session, session_id: str) -> List[Message]:
    """Return all messages for a session ordered chronologically.

    Args:
        db: Active database session.
        session_id: Identifier of the session to load.

    Returns:
        A list of :class:`Message` ordered by ``created_at`` then ``id``.
    """

    stmt = (
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at, Message.id)
    )
    return list(db.scalars(stmt).all())


def reset_session(db: Session, session_id: str) -> int:
    """Delete all messages for a session (keeps the session row).

    Args:
        db: Active database session.
        session_id: Identifier of the session to clear.

    Returns:
        The number of messages that were deleted.
    """

    messages = get_history(db, session_id)
    count = len(messages)
    for message in messages:
        db.delete(message)
    db.commit()
    return count

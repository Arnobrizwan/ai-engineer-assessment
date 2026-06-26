"""Unit tests for the database persistence layer."""

from __future__ import annotations

from app import db as db_module


def test_create_session(db_session) -> None:
    """A session is created on first reference and reused thereafter."""

    s1 = db_module.get_or_create_session(db_session, "abc")
    s2 = db_module.get_or_create_session(db_session, "abc")
    assert s1.id == "abc"
    assert s2.id == "abc"
    assert db_session.get(db_module.ChatSession, "abc") is not None


def test_append_message_creates_session(db_session) -> None:
    """Appending a message auto-creates the parent session."""

    msg = db_module.add_message(db_session, "sess", "user", "hello")
    assert msg.id is not None
    assert msg.role == "user"
    assert msg.content == "hello"
    assert db_session.get(db_module.ChatSession, "sess") is not None


def test_history_ordering(db_session) -> None:
    """History is returned in insertion (chronological) order."""

    db_module.add_message(db_session, "s", "user", "first")
    db_module.add_message(db_session, "s", "assistant", "second")
    db_module.add_message(db_session, "s", "user", "third")

    history = db_module.get_history(db_session, "s")
    assert [m.content for m in history] == ["first", "second", "third"]
    assert [m.role for m in history] == ["user", "assistant", "user"]


def test_history_is_session_scoped(db_session) -> None:
    """Messages from other sessions are not returned."""

    db_module.add_message(db_session, "a", "user", "in-a")
    db_module.add_message(db_session, "b", "user", "in-b")

    assert [m.content for m in db_module.get_history(db_session, "a")] == ["in-a"]
    assert [m.content for m in db_module.get_history(db_session, "b")] == ["in-b"]


def test_reset_session_clears_messages(db_session) -> None:
    """Resetting removes all messages but keeps the session row."""

    db_module.add_message(db_session, "s", "user", "one")
    db_module.add_message(db_session, "s", "assistant", "two")

    deleted = db_module.reset_session(db_session, "s")
    assert deleted == 2
    assert db_module.get_history(db_session, "s") == []
    assert db_session.get(db_module.ChatSession, "s") is not None

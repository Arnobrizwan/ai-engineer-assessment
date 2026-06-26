"""Unit tests for conversation context assembly."""

from __future__ import annotations

from app import db as db_module
from app.main import SYSTEM_PROMPT, build_context


def test_build_context_prepends_system_prompt(db_session) -> None:
    """The assembled context always starts with the system prompt."""

    db_module.add_message(db_session, "s", "user", "hi")
    context = build_context(db_module.get_history(db_session, "s"))

    assert context[0] == {"role": "system", "content": SYSTEM_PROMPT}


def test_build_context_includes_prior_turns_in_order(db_session) -> None:
    """All prior turns are included after the system prompt, in order."""

    db_module.add_message(db_session, "s", "user", "what is 2+2?")
    db_module.add_message(db_session, "s", "assistant", "4")
    db_module.add_message(db_session, "s", "user", "and times 3?")

    context = build_context(db_module.get_history(db_session, "s"))

    assert context == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "what is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "and times 3?"},
    ]


def test_build_context_empty_history() -> None:
    """With no history the context is just the system prompt."""

    assert build_context([]) == [{"role": "system", "content": SYSTEM_PROMPT}]

"""Unit tests for the Ollama streaming client parsing logic.

The HTTP layer (``_iter_ollama_lines``) is monkeypatched so no network or live
model is required: we verify that NDJSON lines are correctly parsed into tokens.
"""

from __future__ import annotations

import json
from typing import AsyncIterator, List

import pytest

from app import llm


def _ndjson_lines() -> List[str]:
    """Return a canned sequence of Ollama NDJSON streaming lines."""

    return [
        json.dumps({"message": {"role": "assistant", "content": "Hel"}, "done": False}),
        json.dumps({"message": {"role": "assistant", "content": "lo"}, "done": False}),
        "not-json-keepalive",  # tolerated and skipped
        json.dumps({"message": {"role": "assistant", "content": ""}, "done": False}),
        json.dumps({"message": {"role": "assistant", "content": "!"}, "done": True}),
    ]


@pytest.mark.asyncio
async def test_stream_chat_parses_tokens(monkeypatch) -> None:
    """stream_chat yields only non-empty content fragments, stopping on done."""

    async def fake_iter(messages, settings) -> AsyncIterator[str]:
        for line in _ndjson_lines():
            yield line

    monkeypatch.setattr(llm, "_iter_ollama_lines", fake_iter)

    tokens = []
    async for token in llm.stream_chat([{"role": "user", "content": "hi"}]):
        tokens.append(token)

    assert tokens == ["Hel", "lo", "!"]


@pytest.mark.asyncio
async def test_stream_chat_stops_after_done(monkeypatch) -> None:
    """Lines after a done marker are not yielded."""

    async def fake_iter(messages, settings) -> AsyncIterator[str]:
        yield json.dumps({"message": {"content": "A"}, "done": True})
        yield json.dumps({"message": {"content": "B"}, "done": False})

    monkeypatch.setattr(llm, "_iter_ollama_lines", fake_iter)

    tokens = [t async for t in llm.stream_chat([{"role": "user", "content": "x"}])]
    assert tokens == ["A"]

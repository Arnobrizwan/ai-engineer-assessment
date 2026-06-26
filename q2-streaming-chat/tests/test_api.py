"""Endpoint tests using FastAPI's TestClient with a mocked Ollama client.

These tests run without any live LLM: ``app.main.stream_chat`` is monkeypatched
with a fake async generator that yields deterministic tokens.
"""

from __future__ import annotations

from typing import AsyncIterator, List

import pytest

import app.main as main


def _fake_stream(tokens: List[str]):
    """Return a fake ``stream_chat`` coroutine generator yielding ``tokens``."""

    async def _gen(messages, settings=None) -> AsyncIterator[str]:
        # Echo nothing about the messages; just stream the canned tokens.
        for token in tokens:
            yield token

    return _gen


def _parse_sse(raw: str) -> List[dict]:
    """Parse an SSE response body into a list of JSON payloads."""

    import json

    payloads = []
    for block in raw.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data:"):
                payloads.append(json.loads(line[len("data:"):].strip()))
    return payloads


def test_chat_streams_tokens(client, monkeypatch) -> None:
    """POST /api/chat streams start/token/done SSE frames."""

    monkeypatch.setattr(main, "stream_chat", _fake_stream(["Hel", "lo", "!"]))

    with client.stream(
        "POST", "/api/chat", json={"message": "hi", "session_id": "t1"}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())

    payloads = _parse_sse(body)
    types = [p["type"] for p in payloads]
    assert types[0] == "start"
    assert types[-1] == "done"

    tokens = [p["content"] for p in payloads if p["type"] == "token"]
    assert tokens == ["Hel", "lo", "!"]

    done = payloads[-1]
    assert done["content"] == "Hello!"


def test_chat_persists_user_and_assistant_messages(client, monkeypatch) -> None:
    """After streaming, both the user message and full reply are persisted."""

    monkeypatch.setattr(main, "stream_chat", _fake_stream(["Four", " is ", "4"]))

    with client.stream(
        "POST", "/api/chat", json={"message": "2+2?", "session_id": "persist"}
    ) as response:
        list(response.iter_text())  # drain the stream

    history = client.get("/api/history", params={"session_id": "persist"}).json()
    messages = history["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "2+2?"
    assert messages[1]["content"] == "Four is 4"


def test_context_includes_prior_turns(client, monkeypatch) -> None:
    """The second request sends prior turns to the LLM (memory)."""

    captured = {}

    async def _capturing_gen(messages, settings=None):
        captured["messages"] = messages
        yield "ok"

    # First turn establishes history.
    monkeypatch.setattr(main, "stream_chat", _fake_stream(["hi there"]))
    with client.stream(
        "POST", "/api/chat", json={"message": "hello", "session_id": "mem"}
    ) as r:
        list(r.iter_text())

    # Second turn captures the context that would be sent to Ollama.
    monkeypatch.setattr(main, "stream_chat", _capturing_gen)
    with client.stream(
        "POST", "/api/chat", json={"message": "again", "session_id": "mem"}
    ) as r:
        list(r.iter_text())

    contents = [m["content"] for m in captured["messages"]]
    # System prompt + first user + first assistant + second user.
    assert "hello" in contents
    assert "hi there" in contents
    assert "again" in contents


def test_history_endpoint_returns_ordered_messages(client, monkeypatch) -> None:
    """GET /api/history returns persisted messages in order."""

    monkeypatch.setattr(main, "stream_chat", _fake_stream(["a", "b"]))
    with client.stream(
        "POST", "/api/chat", json={"message": "first", "session_id": "h"}
    ) as r:
        list(r.iter_text())

    data = client.get("/api/history", params={"session_id": "h"}).json()
    assert data["session_id"] == "h"
    assert data["messages"][0]["content"] == "first"
    assert data["messages"][1]["content"] == "ab"


def test_reset_endpoint_clears_history(client, monkeypatch) -> None:
    """POST /api/session/reset deletes messages for the session."""

    monkeypatch.setattr(main, "stream_chat", _fake_stream(["x"]))
    with client.stream(
        "POST", "/api/chat", json={"message": "hi", "session_id": "r"}
    ) as resp:
        list(resp.iter_text())

    reset = client.post("/api/session/reset", json={"session_id": "r"}).json()
    assert reset["deleted"] == 2

    data = client.get("/api/history", params={"session_id": "r"}).json()
    assert data["messages"] == []


def test_delete_session_alias(client, monkeypatch) -> None:
    """DELETE /api/session also clears history."""

    monkeypatch.setattr(main, "stream_chat", _fake_stream(["y"]))
    with client.stream(
        "POST", "/api/chat", json={"message": "hi", "session_id": "d"}
    ) as resp:
        list(resp.iter_text())

    deleted = client.request(
        "DELETE", "/api/session", params={"session_id": "d"}
    ).json()
    assert deleted["deleted"] == 2


def test_empty_message_rejected(client) -> None:
    """An empty message fails validation with 422."""

    response = client.post("/api/chat", json={"message": "", "session_id": "x"})
    assert response.status_code == 422


def test_healthz(client) -> None:
    """Health probe returns ok status and model name."""

    data = client.get("/healthz").json()
    assert data["status"] == "ok"
    assert "model" in data


@pytest.mark.integration
def test_integration_real_ollama() -> None:
    """Optional end-to-end test against a live Ollama; auto-skips if down."""

    import asyncio

    from app.llm import is_ollama_up, stream_chat

    if not asyncio.run(is_ollama_up()):
        pytest.skip("Ollama is not running; skipping integration test.")

    async def _run() -> str:
        out = []
        messages = [{"role": "user", "content": "Reply with the single word: ping"}]
        async for token in stream_chat(messages):
            out.append(token)
        return "".join(out)

    result = asyncio.run(_run())
    assert isinstance(result, str)
    assert len(result) > 0

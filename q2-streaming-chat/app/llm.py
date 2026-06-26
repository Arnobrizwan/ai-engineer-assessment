"""Ollama streaming client.

This module wraps the Ollama native chat API
(``POST {base}/api/chat`` with ``"stream": true``) and exposes a single async
generator :func:`stream_chat` that yields assistant tokens as plain strings as
they arrive over the wire.

The native Ollama streaming protocol is newline-delimited JSON (NDJSON): each
line is a JSON object of the form::

    {"message": {"role": "assistant", "content": "Hel"}, "done": false}
    {"message": {"role": "assistant", "content": "lo"},  "done": false}
    {"message": {"role": "assistant", "content": ""},    "done": true}

Tests monkeypatch :func:`stream_chat` (or :func:`_iter_ollama_lines`) so the
suite runs without a live model.
"""

from __future__ import annotations

import json
from typing import AsyncIterator, Dict, List

import httpx

from .config import Settings, get_settings

# A "message" as understood by the Ollama chat API: {"role": ..., "content": ...}
ChatMessage = Dict[str, str]


async def _iter_ollama_lines(
    messages: List[ChatMessage], settings: Settings
) -> AsyncIterator[str]:
    """Yield raw NDJSON lines from the Ollama streaming chat endpoint.

    Args:
        messages: Full conversation context in Ollama message format.
        settings: Resolved application settings (base URL, model, timeout).

    Yields:
        Each non-empty line of the streaming HTTP response body.

    Raises:
        httpx.HTTPStatusError: If Ollama returns a non-2xx status.
    """

    url = f"{settings.ollama_base_url}/api/chat"
    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "stream": True,
    }

    timeout = httpx.Timeout(settings.request_timeout, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.strip():
                    yield line


async def stream_chat(
    messages: List[ChatMessage], settings: Settings | None = None
) -> AsyncIterator[str]:
    """Stream assistant tokens for the given conversation.

    Parses each NDJSON line emitted by Ollama and yields the incremental
    ``message.content`` fragment (a "token") as a string. Empty fragments and
    the terminal ``done`` marker are skipped.

    Args:
        messages: Full conversation context (system/user/assistant turns) in
            Ollama message format.
        settings: Optional settings override; defaults to :func:`get_settings`.

    Yields:
        Assistant token fragments in arrival order.
    """

    resolved = settings or get_settings()
    async for line in _iter_ollama_lines(messages, resolved):
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            # Be tolerant of malformed/keep-alive lines.
            continue

        token = chunk.get("message", {}).get("content", "")
        if token:
            yield token

        if chunk.get("done"):
            break


async def is_ollama_up(settings: Settings | None = None) -> bool:
    """Return ``True`` if the Ollama server responds to a health probe.

    Used to auto-skip the optional integration test when no model server is
    running.

    Args:
        settings: Optional settings override.

    Returns:
        ``True`` when the server is reachable, ``False`` otherwise.
    """

    resolved = settings or get_settings()
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{resolved.ollama_base_url}/api/tags")
            return response.status_code == 200
    except (httpx.HTTPError, OSError):
        return False

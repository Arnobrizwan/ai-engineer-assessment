"""Ollama chat client with tool-calling support.

This module wraps Ollama's ``/api/chat`` endpoint (via the official ``ollama``
Python client) and normalises its response into a small, typed structure the
agent loop can consume. Keeping the HTTP/JSON details here means the loop can be
fully unit-tested by swapping in a fake client that satisfies
:class:`LLMClient`'s protocol.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol

import ollama

from .config import Settings, settings as default_settings


@dataclass
class ToolCall:
    """A single tool invocation requested by the model.

    Attributes:
        name: Tool name the model wants to call.
        arguments: Keyword arguments for the tool.
        call_id: Optional provider-supplied identifier for the call.
    """

    name: str
    arguments: dict[str, Any]
    call_id: str | None = None


@dataclass
class LLMResponse:
    """Normalised model turn.

    Attributes:
        content: Assistant text content (may be empty when tool calls present).
        tool_calls: Tool calls requested by the model this turn.
        raw: The raw provider message dict, for debugging.
    """

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        """Whether the model requested at least one tool call."""
        return bool(self.tool_calls)


class LLMClient(Protocol):
    """Protocol implemented by any chat backend the agent loop can drive."""

    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        """Send ``messages`` (+ tool schemas) and return a normalised response."""
        ...


def _normalise_tool_calls(message: dict[str, Any]) -> list[ToolCall]:
    """Extract :class:`ToolCall` objects from an Ollama assistant message."""
    calls: list[ToolCall] = []
    for raw_call in message.get("tool_calls") or []:
        function = raw_call.get("function", {}) if isinstance(raw_call, dict) else {}
        name = function.get("name", "")
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            # Some providers return arguments as a JSON string.
            import json

            try:
                arguments = json.loads(arguments) if arguments else {}
            except json.JSONDecodeError:
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        calls.append(
            ToolCall(
                name=name,
                arguments=arguments,
                call_id=raw_call.get("id") if isinstance(raw_call, dict) else None,
            )
        )
    return calls


_FENCE_RE = re.compile(r"```(?:json|tool_call|python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _iter_json_objects(text: str) -> Iterable[str]:
    """Yield top-level ``{...}`` substrings from ``text`` via brace matching.

    The scan is string-literal aware so braces/quotes inside JSON string values
    (or inside a SQL query argument) do not confuse the matcher.

    Args:
        text: Arbitrary text that may contain one or more JSON objects.

    Yields:
        Each balanced ``{...}`` substring, in order of appearance.
    """
    depth = 0
    start: int | None = None
    in_str = False
    escaped = False
    for index, char in enumerate(text):
        if in_str:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_str = False
            continue
        if char == '"':
            in_str = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    yield text[start : index + 1]
                    start = None


def parse_inline_tool_calls(
    content: str, valid_names: Iterable[str]
) -> list[ToolCall]:
    """Recover tool calls that a model emitted as inline text in ``content``.

    Many local models (notably ``llama3.1``) do not always populate Ollama's
    native ``tool_calls`` field; instead they write the call as a JSON object in
    the assistant message body, e.g.::

        It seems I should look at the tables first.
        {"name": "list_tables", "parameters": {}}

    or wrapped in a ```json fenced block. This function extracts such objects
    (whether fenced, surrounded by prose, or a bare object) and returns the ones
    whose ``name`` matches a known tool. The argument object may be keyed as
    ``parameters`` or ``arguments``.

    Args:
        content: The assistant message text to scan.
        valid_names: Iterable of recognised tool names; others are ignored.

    Returns:
        A list of :class:`ToolCall` objects (possibly empty), de-duplicated by
        the raw JSON substring they were parsed from.
    """
    if not content:
        return []

    names = set(valid_names)
    # Scan fenced blocks first (highest signal), then the whole body so bare
    # objects surrounded by prose are still caught. De-dup identical substrings.
    segments: list[str] = list(_FENCE_RE.findall(content))
    segments.append(content)

    calls: list[ToolCall] = []
    seen: set[str] = set()
    for segment in segments:
        for raw in _iter_json_objects(segment):
            if raw in seen:
                continue
            seen.add(raw)
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            name = obj.get("name")
            if not isinstance(name, str) or name not in names:
                continue
            arguments = obj.get("parameters")
            if arguments is None:
                arguments = obj.get("arguments", {})
            if not isinstance(arguments, dict):
                arguments = {}
            calls.append(ToolCall(name=name, arguments=arguments))
    return calls


class OllamaClient:
    """Concrete :class:`LLMClient` backed by a local Ollama server.

    Args:
        settings: Optional settings override; defaults to module settings.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or default_settings
        self._client = ollama.Client(
            host=self._settings.ollama_base_url,
            timeout=self._settings.request_timeout,
        )

    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        """Call Ollama ``/api/chat`` with tool schemas and normalise the result.

        Args:
            messages: Conversation so far in Ollama message format.
            tools: Tool JSON schemas to expose to the model.

        Returns:
            A normalised :class:`LLMResponse`.
        """
        response = self._client.chat(
            model=self._settings.llm_model,
            messages=messages,
            tools=tools,
            options={"temperature": 0.0},
        )
        # The ollama client (>=0.4) returns Pydantic models, not plain dicts.
        # Convert the whole response to nested plain dicts so the rest of the
        # pipeline (and the inline-JSON fallback) sees a uniform structure.
        if hasattr(response, "model_dump"):
            response_dict: dict[str, Any] = response.model_dump()
        else:  # pragma: no cover - older client returning a mapping
            response_dict = dict(response)

        message = response_dict.get("message") or {}
        if not isinstance(message, dict):  # pragma: no cover - defensive
            message = dict(message)
        return LLMResponse(
            content=message.get("content", "") or "",
            tool_calls=_normalise_tool_calls(message),
            raw=message,
        )


def is_ollama_available(settings: Settings | None = None) -> bool:
    """Best-effort check that the configured Ollama server is reachable.

    Args:
        settings: Optional settings override.

    Returns:
        ``True`` if the server responds to a list request, ``False`` otherwise.
    """
    cfg = settings or default_settings
    try:
        client = ollama.Client(host=cfg.ollama_base_url, timeout=3.0)
        client.list()
        return True
    except Exception:  # noqa: BLE001 - any failure means "unavailable"
        return False

"""Tests for the agent loop control flow using a mocked LLM.

These tests run WITHOUT any live LLM by driving the :class:`Agent` with a
scripted fake client that returns pre-baked :class:`LLMResponse` objects. This
lets us assert the loop:

* calls the right tools in order,
* terminates when the model stops requesting tools,
* self-corrects after a bad SQL query, and
* respects the iteration cap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from agent.config import Settings
from agent.llm import LLMResponse, ToolCall, parse_inline_tool_calls
from agent.loop import Agent, StepType
from agent.tools import Toolbox, tool_names


def _settings(max_iterations: int = 8) -> Settings:
    """Build a Settings instance for tests (db_path unused by the loop)."""
    from pathlib import Path

    return Settings(
        ollama_base_url="http://localhost:11434",
        llm_model="test-model",
        database_url="sqlite:///./test.db",
        db_path=Path("./test.db"),
        max_iterations=max_iterations,
        sql_row_limit=1000,
        request_timeout=5.0,
    )


def _tool_response(name: str, arguments: dict[str, Any]) -> LLMResponse:
    """Helper: an assistant turn that requests a single tool call."""
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(name=name, arguments=arguments)],
        raw={
            "tool_calls": [
                {"function": {"name": name, "arguments": arguments}}
            ]
        },
    )


def _final_response(text: str) -> LLMResponse:
    """Helper: a final assistant turn with no tool calls."""
    return LLMResponse(content=text, tool_calls=[], raw={})


def _inline_response(text: str) -> LLMResponse:
    """Helper: an assistant turn with NO native tool_calls, only inline text.

    Mirrors llama3.1's common behaviour of emitting the tool call as JSON inside
    the message body instead of the structured ``tool_calls`` field.
    """
    return LLMResponse(content=text, tool_calls=[], raw={})


@dataclass
class ScriptedLLM:
    """A fake :class:`LLMClient` that replays a fixed list of responses.

    Attributes:
        responses: The queued responses, returned in order on each ``chat``.
        calls: Records every (messages, tools) pair the loop sent, for asserts.
    """

    responses: list[LLMResponse]
    calls: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = field(
        default_factory=list
    )

    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        self.calls.append((list(messages), tools))
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[index]


def test_happy_path_calls_tool_then_answers(toolbox: Toolbox) -> None:
    """Model runs one SQL query then produces a final answer."""
    llm = ScriptedLLM(
        [
            _tool_response("run_sql", {"query": "SELECT COUNT(*) AS n FROM customers"}),
            _final_response("There are 8 customers."),
        ]
    )
    agent = Agent(llm, toolbox, _settings())
    result = agent.run("How many customers are there?")

    assert result.completed is True
    assert result.answer == "There are 8 customers."
    assert result.iterations == 2
    assert result.last_sql == "SELECT COUNT(*) AS n FROM customers"
    assert result.last_result is not None
    assert result.last_result["rows"][0][0] == 8

    # Loop made exactly two LLM calls.
    assert len(llm.calls) == 2

    # Trace contains a tool call followed by an observation and a final answer.
    types = [s.step_type for s in result.trace]
    assert StepType.TOOL_CALL in types
    assert StepType.OBSERVATION in types
    assert types[-1] == StepType.FINAL


def test_schema_then_query_flow(toolbox: Toolbox) -> None:
    """Model introspects schema, queries, then answers."""
    llm = ScriptedLLM(
        [
            _tool_response("list_tables", {}),
            _tool_response("get_schema", {"table": "products"}),
            _tool_response(
                "run_sql", {"query": "SELECT MAX(price) AS m FROM products"}
            ),
            _final_response("The most expensive product costs 450."),
        ]
    )
    agent = Agent(llm, toolbox, _settings())
    result = agent.run("What is the priciest product?")

    assert result.completed is True
    tool_calls = [s.tool_name for s in result.trace if s.step_type == StepType.TOOL_CALL]
    assert tool_calls == ["list_tables", "get_schema", "run_sql"]


def test_self_correction_on_bad_sql(toolbox: Toolbox) -> None:
    """Model emits a bad query, sees the error, then a corrected query."""
    bad = _tool_response("run_sql", {"query": "SELECT nope FROM customers"})
    good = _tool_response(
        "run_sql", {"query": "SELECT COUNT(*) AS n FROM customers"}
    )
    final = _final_response("There are 8 customers.")
    llm = ScriptedLLM([bad, good, final])

    agent = Agent(llm, toolbox, _settings())
    result = agent.run("How many customers?")

    assert result.completed is True
    # The error observation must have been recorded before the good result.
    observations = [
        s for s in result.trace if s.step_type == StepType.OBSERVATION
    ]
    assert "error" in (observations[0].observation or {})
    assert "error" not in (observations[1].observation or {})
    # Final tracked SQL is the corrected one.
    assert result.last_sql == "SELECT COUNT(*) AS n FROM customers"

    # The error payload was fed back to the model as a tool message.
    second_call_messages = llm.calls[1][0]
    assert any(m.get("role") == "tool" for m in second_call_messages)


def test_guard_blocks_dml_inside_loop(toolbox: Toolbox) -> None:
    """A DML query requested by the model is rejected, model then answers."""
    llm = ScriptedLLM(
        [
            _tool_response("run_sql", {"query": "DELETE FROM customers"}),
            _final_response("I cannot modify data; here is a read-only summary."),
        ]
    )
    agent = Agent(llm, toolbox, _settings())
    result = agent.run("Delete all customers")

    observation = next(
        s for s in result.trace if s.step_type == StepType.OBSERVATION
    )
    assert "error" in (observation.observation or {})
    assert result.completed is True


def test_iteration_cap_enforced(toolbox: Toolbox) -> None:
    """If the model never stops calling tools, the loop terminates at the cap."""
    looping = _tool_response("list_tables", {})
    llm = ScriptedLLM([looping])  # always returns a tool call
    agent = Agent(llm, toolbox, _settings(max_iterations=3))
    result = agent.run("loop forever")

    assert result.completed is False
    assert result.iterations == 3
    assert any(s.step_type == StepType.ERROR for s in result.trace)


def test_inline_tool_call_in_content_is_parsed(toolbox: Toolbox) -> None:
    """An inline JSON tool call (no native tool_calls) must be executed.

    Regression guard for llama3.1's habit of writing the tool call as text in
    ``message.content`` rather than the native ``tool_calls`` field. The loop
    must NOT treat this as a final answer.
    """
    inline = _inline_response(
        "It seems like I need to call list_tables first to see the tables.\n"
        '{"name": "run_sql", "parameters": {"query": "SELECT COUNT(*) AS n FROM customers"}}'
    )
    final = _final_response("There are 8 customers.")
    llm = ScriptedLLM([inline, final])

    agent = Agent(llm, toolbox, _settings())
    result = agent.run("How many customers are there?")

    assert result.completed is True
    # The inline call was executed, not treated as the final answer.
    assert result.last_sql == "SELECT COUNT(*) AS n FROM customers"
    assert result.last_result is not None
    assert result.last_result["rows"][0][0] == 8
    # The inline turn produced a real tool call + observation in the trace.
    tool_calls = [
        s.tool_name for s in result.trace if s.step_type == StepType.TOOL_CALL
    ]
    assert "run_sql" in tool_calls
    assert result.iterations == 2


def test_inline_tool_call_fenced_json_is_parsed(toolbox: Toolbox) -> None:
    """A fenced ```json tool call wrapped in prose is parsed and executed."""
    inline = _inline_response(
        "Let me check the available tables.\n"
        "```json\n"
        '{"name": "list_tables", "arguments": {}}\n'
        "```"
    )
    final = _final_response("There are four tables.")
    llm = ScriptedLLM([inline, final])

    agent = Agent(llm, toolbox, _settings())
    result = agent.run("What tables exist?")

    assert result.completed is True
    tool_calls = [
        s.tool_name for s in result.trace if s.step_type == StepType.TOOL_CALL
    ]
    assert tool_calls == ["list_tables"]


def test_inline_unknown_tool_treated_as_answer(toolbox: Toolbox) -> None:
    """Inline JSON that is not a known tool is treated as a final answer."""
    inline = _inline_response('Here is some JSON: {"foo": "bar", "name": "nope"}')
    llm = ScriptedLLM([inline])
    agent = Agent(llm, toolbox, _settings())
    result = agent.run("anything")

    assert result.completed is True
    assert result.iterations == 1
    assert "foo" in result.answer


# -- direct parser-level unit tests (no agent, no network) --


def test_parser_bare_object_with_parameters() -> None:
    names = set(tool_names())
    calls = parse_inline_tool_calls(
        'prose {"name": "get_schema", "parameters": {"table": "orders"}} more',
        names,
    )
    assert len(calls) == 1
    assert calls[0].name == "get_schema"
    assert calls[0].arguments == {"table": "orders"}


def test_parser_arguments_key_and_sql_with_braces() -> None:
    names = set(tool_names())
    # SQL value containing a brace-ish string should not break brace matching.
    text = '{"name": "run_sql", "arguments": {"query": "SELECT 1 WHERE a = \'{x}\'"}}'
    calls = parse_inline_tool_calls(text, names)
    assert len(calls) == 1
    assert calls[0].name == "run_sql"
    assert "SELECT 1" in calls[0].arguments["query"]


def test_parser_ignores_unknown_and_malformed() -> None:
    names = set(tool_names())
    assert parse_inline_tool_calls("no json here", names) == []
    assert parse_inline_tool_calls('{"name": "unknown_tool"}', names) == []
    assert parse_inline_tool_calls('{"name": broken', names) == []
    assert parse_inline_tool_calls("", names) == []

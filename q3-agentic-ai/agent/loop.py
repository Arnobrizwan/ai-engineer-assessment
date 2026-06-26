"""The agentic tool-calling loop (ReAct / function-calling style).

The loop is the orchestrator that gives the system its *agency*:

    plan  ->  call tool  ->  observe result  ->  re-plan  ->  ... -> final answer

On each iteration the LLM (the "brain") either requests one or more tool calls
or produces a final natural-language answer. The loop executes requested tools
against the real environment (the SQLite database via :class:`~agent.tools.Toolbox`),
feeds the observations back as ``tool`` messages, and repeats until the model
answers or a hard iteration cap is hit. SQL errors are returned to the model so
it can **self-correct** on the following turn.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .config import Settings, settings as default_settings
from .llm import LLMClient, LLMResponse, parse_inline_tool_calls
from .tools import TOOL_SCHEMAS, Toolbox, tool_names

SYSTEM_PROMPT = """\
You are a meticulous SQL data analyst agent. You answer business questions by \
querying a read-only SQLite analytics database using the provided tools.

Follow this process:
1. If you are unsure of the tables, call list_tables.
2. Before writing SQL, call get_schema on the relevant table(s) to confirm the \
exact column names.
3. Use run_sql to execute a single read-only SELECT. Only SELECT is allowed.
4. If run_sql returns an error, read the error message carefully and try a \
corrected query. Do not repeat the same failing query.
5. Optionally call make_chart to visualise a result.
6. When you have enough grounded information, stop calling tools and write a \
clear, concise final answer that directly answers the user's question and cites \
the numbers you found.

You MUST use the tools to gather data before answering; do not guess. Prefer the \
native tool-calling interface. If for any reason you cannot use native tool \
calls, then emit ONLY a single JSON object on its own line of the exact form \
{"name": "<tool_name>", "parameters": { ... }} and nothing else, so it can be \
parsed and executed. Do not wrap it in extra commentary when doing so.

Never fabricate data. Base every number in your answer on a tool observation."""


class StepType(str, Enum):
    """Kinds of entries recorded in the agent trace."""

    THOUGHT = "thought"
    TOOL_CALL = "tool_call"
    OBSERVATION = "observation"
    FINAL = "final"
    ERROR = "error"


@dataclass
class TraceStep:
    """A single recorded step in the agent's reasoning trace.

    Attributes:
        step_type: The category of step (see :class:`StepType`).
        content: Human-readable text for the step (e.g. a thought or answer).
        tool_name: Tool name, for ``TOOL_CALL`` / ``OBSERVATION`` steps.
        tool_args: Tool arguments, for ``TOOL_CALL`` steps.
        observation: Structured tool result, for ``OBSERVATION`` steps.
        iteration: The loop iteration this step belongs to (1-indexed).
    """

    step_type: StepType
    content: str = ""
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    observation: dict[str, Any] | None = None
    iteration: int = 0


@dataclass
class AgentResult:
    """The outcome of an :class:`Agent.run` invocation.

    Attributes:
        question: The original user question.
        answer: The final natural-language answer (empty if not reached).
        trace: Ordered list of every step taken.
        iterations: Number of loop iterations executed.
        completed: Whether the agent produced a final answer within the cap.
        last_sql: The most recent SQL submitted to ``run_sql`` (if any).
        last_result: The most recent successful SQL result payload (if any).
        last_chart: The most recent chart specification produced (if any).
    """

    question: str
    answer: str
    trace: list[TraceStep] = field(default_factory=list)
    iterations: int = 0
    completed: bool = False
    last_sql: str | None = None
    last_result: dict[str, Any] | None = None
    last_chart: dict[str, Any] | None = None


def _summarise_observation(name: str, result: dict[str, Any]) -> str:
    """Produce a short human-readable summary of a tool observation."""
    if "error" in result:
        return f"{name} error: {result['error']}"
    if name == "list_tables":
        return f"tables: {', '.join(result.get('tables', []))}"
    if name == "get_schema":
        cols = ", ".join(c["name"] for c in result.get("columns", []))
        return f"schema({result.get('table')}): {cols}"
    if name == "run_sql":
        return f"{result.get('row_count', 0)} row(s) returned"
    if name == "make_chart":
        return f"{result.get('chart_type')} chart with {len(result.get('values', []))} points"
    return "ok"


class Agent:
    """Drives the tool-calling loop against an :class:`LLMClient` + environment.

    Args:
        llm: Any object satisfying the :class:`~agent.llm.LLMClient` protocol.
        toolbox: The bound :class:`~agent.tools.Toolbox` (the environment).
        settings: Optional settings override (used for the iteration cap).
        system_prompt: Optional system prompt override.
    """

    def __init__(
        self,
        llm: LLMClient,
        toolbox: Toolbox,
        settings: Settings | None = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self._llm = llm
        self._toolbox = toolbox
        self._settings = settings or default_settings
        self._system_prompt = system_prompt
        self._valid_tool_names = set(tool_names())

    def run(self, question: str) -> AgentResult:
        """Answer ``question`` by planning, calling tools, and synthesising.

        Args:
            question: The natural-language business question.

        Returns:
            A populated :class:`AgentResult` including the full trace.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": question},
        ]
        result = AgentResult(question=question, answer="")

        max_iterations = max(1, self._settings.max_iterations)
        for iteration in range(1, max_iterations + 1):
            result.iterations = iteration
            response: LLMResponse = self._llm.chat(messages, TOOL_SCHEMAS)

            # Prefer Ollama's native structured tool_calls. Many local models
            # (e.g. llama3.1) instead embed the call as inline JSON in the
            # message body, so fall back to parsing it out of the content.
            tool_calls = response.tool_calls
            if not tool_calls and response.content.strip():
                tool_calls = parse_inline_tool_calls(
                    response.content, self._valid_tool_names
                )
            has_tool_calls = bool(tool_calls)

            # Record any narrative "thought" the model emitted this turn.
            if response.content.strip():
                step_type = (
                    StepType.FINAL if not has_tool_calls else StepType.THOUGHT
                )
                result.trace.append(
                    TraceStep(
                        step_type=step_type,
                        content=response.content.strip(),
                        iteration=iteration,
                    )
                )

            # No parseable tool call => the model has produced its final answer.
            if not has_tool_calls:
                result.answer = response.content.strip()
                result.completed = True
                return result

            # Append the assistant message (with any native tool calls) to the
            # history so the model sees its own turn before the observations.
            messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": response.raw.get("tool_calls", []),
                }
            )

            # Execute every requested tool call and feed observations back.
            for call in tool_calls:
                result.trace.append(
                    TraceStep(
                        step_type=StepType.TOOL_CALL,
                        content=f"Calling {call.name}",
                        tool_name=call.name,
                        tool_args=call.arguments,
                        iteration=iteration,
                    )
                )
                observation = self._toolbox.call(call.name, call.arguments)

                # Track useful artefacts for the UI / callers.
                if call.name == "run_sql":
                    result.last_sql = call.arguments.get("query")
                    if "error" not in observation:
                        result.last_result = observation
                elif call.name == "make_chart" and "error" not in observation:
                    result.last_chart = observation

                result.trace.append(
                    TraceStep(
                        step_type=StepType.OBSERVATION,
                        content=_summarise_observation(call.name, observation),
                        tool_name=call.name,
                        observation=observation,
                        iteration=iteration,
                    )
                )
                messages.append(
                    {
                        "role": "tool",
                        "name": call.name,
                        "content": json.dumps(observation),
                    }
                )

        # Hit the iteration cap without a final answer.
        result.completed = False
        result.answer = (
            "I reached the maximum number of reasoning steps without producing a "
            "final answer. The partial findings are available in the trace above."
        )
        result.trace.append(
            TraceStep(
                step_type=StepType.ERROR,
                content="Iteration cap reached without a final answer.",
                iteration=result.iterations,
            )
        )
        return result

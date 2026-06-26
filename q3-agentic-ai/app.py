"""Streamlit prototype for the SQL Analytics Agent.

Run with:
    streamlit run app.py

The UI lets a user ask a natural-language business question and then shows the
agent's full reasoning trace (thought -> tool call -> observation), the
generated SQL, the result table, an optional chart, and the final answer.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from agent.config import get_settings
from agent.db import ensure_seeded
from agent.llm import OllamaClient, is_ollama_available
from agent.loop import Agent, AgentResult, StepType, TraceStep
from agent.tools import Toolbox

st.set_page_config(page_title="SQL Analytics Agent", page_icon="📊", layout="wide")


@st.cache_resource
def _bootstrap() -> tuple[Toolbox, Any]:
    """Seed the database (if needed) and build shared resources.

    Returns:
        A tuple of (toolbox, settings).
    """
    settings = get_settings()
    ensure_seeded(settings.db_path)
    toolbox = Toolbox(settings.db_path, row_limit=settings.sql_row_limit)
    return toolbox, settings


def _render_observation(step: TraceStep) -> None:
    """Render a single observation step (SQL table, chart, or raw payload)."""
    obs = step.observation or {}
    if "error" in obs:
        st.error(obs["error"])
        return
    if step.tool_name == "run_sql":
        columns = obs.get("columns", [])
        rows = obs.get("rows", [])
        if columns:
            st.dataframe(pd.DataFrame(rows, columns=columns), use_container_width=True)
        if obs.get("truncated"):
            st.caption("Result truncated to the configured row limit.")
    elif step.tool_name == "make_chart":
        _render_chart(obs)
    else:
        st.json(obs)


def _render_chart(spec: dict[str, Any]) -> None:
    """Render a chart specification produced by the ``make_chart`` tool."""
    labels = spec.get("labels", [])
    values = spec.get("values", [])
    if not labels or not values:
        return
    frame = pd.DataFrame({"label": labels, "value": values}).set_index("label")
    if spec.get("title"):
        st.caption(spec["title"])
    if spec.get("chart_type") == "line":
        st.line_chart(frame)
    else:
        st.bar_chart(frame)


def _render_trace(result: AgentResult) -> None:
    """Render the full step-by-step agent trace."""
    icons = {
        StepType.THOUGHT: "🧠",
        StepType.TOOL_CALL: "🔧",
        StepType.OBSERVATION: "👁️",
        StepType.FINAL: "✅",
        StepType.ERROR: "⚠️",
    }
    for step in result.trace:
        icon = icons.get(step.step_type, "•")
        if step.step_type == StepType.TOOL_CALL:
            with st.expander(f"{icon} Tool call: `{step.tool_name}`", expanded=False):
                st.json(step.tool_args or {})
        elif step.step_type == StepType.OBSERVATION:
            with st.expander(
                f"{icon} Observation from `{step.tool_name}` — {step.content}",
                expanded=(step.tool_name == "run_sql"),
            ):
                _render_observation(step)
        elif step.step_type == StepType.THOUGHT:
            st.markdown(f"{icon} **Thought:** {step.content}")
        elif step.step_type == StepType.ERROR:
            st.warning(f"{icon} {step.content}")


def main() -> None:
    """Streamlit app entry point."""
    toolbox, settings = _bootstrap()

    st.title("📊 SQL Analytics Agent")
    st.caption(
        "Ask a business question in plain English. The agent plans, calls "
        "read-only SQL tools against a seeded e-commerce database, self-corrects "
        "on errors, and synthesises a grounded answer."
    )

    with st.sidebar:
        st.header("Status")
        ollama_up = is_ollama_available(settings)
        if ollama_up:
            st.success(f"Ollama reachable · model `{settings.llm_model}`")
        else:
            st.error(
                "Ollama not reachable. Start it with `ollama serve` and pull the "
                f"model: `ollama pull {settings.llm_model}`."
            )
        st.write("**Database**")
        st.code(str(settings.db_path), language="text")
        st.write("**Tables**")
        st.write(", ".join(toolbox.list_tables().get("tables", [])))
        st.write(f"**Max iterations:** {settings.max_iterations}")
        st.divider()
        st.markdown(
            "**Examples**\n\n"
            "- Which product generated the most revenue?\n"
            "- How many completed orders did each country place?\n"
            "- What is the average order value?\n"
            "- Show monthly revenue as a chart."
        )

    question = st.text_input(
        "Your question",
        value="Which product generated the most revenue from completed orders?",
    )
    run_clicked = st.button("Ask the agent", type="primary", disabled=not ollama_up)

    if run_clicked and question.strip():
        agent = Agent(OllamaClient(settings), toolbox, settings)
        with st.spinner("Agent is reasoning..."):
            result = agent.run(question.strip())

        st.subheader("Final answer")
        if result.completed:
            st.success(result.answer)
        else:
            st.warning(result.answer)

        if result.last_chart:
            st.subheader("Chart")
            _render_chart(result.last_chart)

        if result.last_sql:
            st.subheader("Generated SQL")
            st.code(result.last_sql, language="sql")

        st.subheader(f"Reasoning trace ({result.iterations} iteration(s))")
        _render_trace(result)


if __name__ == "__main__":
    main()

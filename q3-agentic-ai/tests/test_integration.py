"""Optional end-to-end integration test against a live Ollama server.

Marked ``integration`` and auto-skipped when Ollama is unreachable or the
configured model is not pulled, so the default ``pytest`` run never requires a
live LLM.

Run explicitly with:
    pytest -m integration
"""

from __future__ import annotations

import pytest

from agent.config import get_settings
from agent.llm import OllamaClient, is_ollama_available
from agent.loop import Agent
from agent.tools import Toolbox

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    not is_ollama_available(get_settings()),
    reason="Ollama server not reachable; skipping live integration test.",
)
def test_live_agent_answers_question(seeded_db) -> None:
    settings = get_settings()
    toolbox = Toolbox(seeded_db, row_limit=settings.sql_row_limit)
    agent = Agent(OllamaClient(settings), toolbox, settings)

    result = agent.run("How many customers are in the database?")

    assert result.iterations >= 1
    # The seeded dataset has 8 customers; a grounded answer should mention it.
    assert "8" in result.answer or result.last_result is not None

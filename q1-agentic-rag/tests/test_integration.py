"""Live integration test -- auto-skipped unless Ollama is reachable.

Run explicitly with: ``pytest -m integration``.
"""

from __future__ import annotations

import pytest

from rag.config import get_config
from rag.llm import LLMClient
from rag.pipeline import build_agent_from_dir

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def live_llm():
    client = LLMClient(get_config())
    if not client.is_available():
        pytest.skip("Ollama server is not reachable; skipping integration test.")
    return client


def test_end_to_end_against_sample_docs(live_llm):
    agent = build_agent_from_dir(llm=live_llm, embed_fn=live_llm.embed)
    result = agent.run("How many moons does Mars have and what are they called?")
    assert result.answer
    assert result.citations, "a grounded answer should contain citations"
    text = result.answer.lower()
    assert "phobos" in text or "deimos" in text or "two" in text

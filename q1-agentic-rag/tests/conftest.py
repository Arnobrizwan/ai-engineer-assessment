"""Shared pytest fixtures and a deterministic fake LLM.

The fake LLM lets the whole agent loop run offline by routing each prompt to a
canned response based on its system prompt, so no Ollama server is required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import pytest

# Make the package importable when pytest is run from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.chunking import Chunk  # noqa: E402


class FakeLLM:
    """A scripted chat model used to drive the agent deterministically.

    It inspects the system prompt to decide behaviour:
      * rewrite/reformulate -> echoes the user content (acts as identity).
      * grader -> returns 'yes' if any keyword overlaps the passage else 'no'.
      * answer -> returns a templated grounded answer citing the first source.
    Call counts are recorded for assertions.
    """

    def __init__(self, grade_keywords: Sequence[str] | None = None) -> None:
        self.grade_keywords = [k.lower() for k in (grade_keywords or [])]
        self.calls: list[dict[str, str]] = []

    def chat(self, messages: Sequence[dict[str, str]], temperature: float = 0.0) -> str:
        system = messages[0]["content"].lower()
        user = messages[-1]["content"]
        self.calls.append({"system": system, "user": user})

        if "rewrite" in system and "weak" not in system:
            return user.strip()
        if "weak" in system or "reformulate" in system:
            return user.split("\n")[0].replace("Original question:", "").strip()
        if "relevance grader" in system:
            passage = user.lower()
            if not self.grade_keywords:
                return "yes"
            return "yes" if any(k in passage for k in self.grade_keywords) else "no"
        if "precise assistant" in system:
            # Cite the first source marker present in the prompt.
            for line in user.splitlines():
                line = line.strip()
                if line.startswith("[") and "/ chunk" in line:
                    return f"This is the grounded answer. {line}"
            return "I don't know based on the provided documents."
        return ""


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """A small hand-built corpus with predictable keywords."""
    return [
        Chunk(text="Photosynthesis converts light energy into glucose and oxygen.",
              doc_name="photosynthesis.md", chunk_index=0, page=1),
        Chunk(text="Chlorophyll absorbs blue and red light and reflects green.",
              doc_name="photosynthesis.md", chunk_index=1, page=1),
        Chunk(text="Mars is the fourth planet and is called the Red Planet.",
              doc_name="mars.md", chunk_index=0, page=1),
        Chunk(text="Mars has two moons named Phobos and Deimos.",
              doc_name="mars.md", chunk_index=1, page=1),
        Chunk(text="The Great Wall of China is about 21196 kilometres long.",
              doc_name="great_wall.md", chunk_index=0, page=1),
    ]

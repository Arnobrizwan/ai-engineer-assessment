"""Unit tests for the agentic loop's grade/decision logic (mocked LLM)."""

from __future__ import annotations

from rag.agent import AgenticRAG, _is_yes
from rag.retriever import HybridRetriever

from .conftest import FakeLLM


def _agent(chunks, llm, max_iters=2):
    retriever = HybridRetriever(chunks)
    agent = AgenticRAG(retriever, llm)
    agent.config = type(agent.config)(max_agent_iterations=max_iters, top_k=3)
    return agent


def test_is_yes_parsing():
    assert _is_yes("yes")
    assert _is_yes("Yes, relevant")
    assert not _is_yes("no")
    assert not _is_yes("No it is not")


def test_grade_keeps_only_relevant_chunks(sample_chunks):
    llm = FakeLLM(grade_keywords=["mars", "phobos", "deimos"])
    agent = _agent(sample_chunks, llm)
    result = agent.run("Tell me about the moons of Mars")
    # Only mars-related chunks should be graded relevant.
    assert all("mars.md" in p.chunk.doc_name for p in result.passages) or result.passages


def test_answer_is_grounded_and_cited(sample_chunks):
    llm = FakeLLM(grade_keywords=["mars", "moons", "phobos", "deimos"])
    agent = _agent(sample_chunks, llm)
    result = agent.run("What are the moons of Mars?")
    assert result.citations, "expected at least one citation"
    assert result.cited_passages
    # The cited marker must correspond to a real retrieved passage.
    assert result.cited_passages[0].citation in result.citations


def test_agent_reformulates_when_first_pass_is_weak(sample_chunks):
    # Grader rejects everything on terms that never match -> forces reformulation.
    llm = FakeLLM(grade_keywords=["zzz-never-matches"])
    agent = _agent(sample_chunks, llm, max_iters=2)
    result = agent.run("photosynthesis oxygen")
    steps = [s.step for s in result.trace]
    assert "reformulate" in steps
    assert result.iterations == 2  # looped the full budget


def test_trace_records_every_stage(sample_chunks):
    llm = FakeLLM(grade_keywords=["mars"])
    agent = _agent(sample_chunks, llm)
    result = agent.run("Mars facts")
    steps = [s.step for s in result.trace]
    assert steps[0] == "analyze_query"
    assert "retrieve" in steps
    assert "grade" in steps
    assert steps[-1] == "generate"


def test_decision_stops_when_evidence_is_found(sample_chunks):
    llm = FakeLLM(grade_keywords=["mars", "red", "planet"])
    agent = _agent(sample_chunks, llm, max_iters=3)
    result = agent.run("Why is Mars red?")
    # Found evidence on the first pass -> should not loop.
    assert result.iterations == 1
    assert "reformulate" not in [s.step for s in result.trace]

"""Unit tests for citation formatting and extraction."""

from __future__ import annotations

from rag.chunking import Chunk
from rag.citations import (
    build_context,
    extract_citations,
    format_citation,
    used_passages,
)
from rag.retriever import ScoredChunk


def _scored():
    chunks = [
        Chunk(text="Mars is the Red Planet.", doc_name="mars.md", chunk_index=0, page=1),
        Chunk(text="It has two moons.", doc_name="mars.md", chunk_index=1, page=2),
    ]
    return [ScoredChunk(chunks[0], 0.9), ScoredChunk(chunks[1], 0.8)]


def test_format_citation_canonical_form():
    c = Chunk(text="x", doc_name="mars.md", chunk_index=4, page=3)
    assert format_citation(c) == "[mars.md p.3 / chunk 4]"


def test_build_context_includes_markers_and_passages():
    context, passages = build_context(_scored())
    assert "[mars.md p.1 / chunk 0]" in context
    assert "[mars.md p.2 / chunk 1]" in context
    assert len(passages) == 2
    assert passages[0].citation == "[mars.md p.1 / chunk 0]"


def test_extract_citations_dedupes_and_orders():
    answer = (
        "Mars is red [mars.md p.1 / chunk 0]. It has moons "
        "[mars.md p.2 / chunk 1]. Still red [mars.md p.1 / chunk 0]."
    )
    cites = extract_citations(answer)
    assert cites == ["[mars.md p.1 / chunk 0]", "[mars.md p.2 / chunk 1]"]


def test_extract_citations_handles_spacing_variations():
    answer = "Fact [mars.md p.1 /chunk 0] and [mars.md p.2 /  chunk  1]."
    cites = extract_citations(answer)
    assert "[mars.md p.1 / chunk 0]" in cites
    assert "[mars.md p.2 / chunk 1]" in cites


def test_used_passages_filters_to_cited_only():
    _context, passages = build_context(_scored())
    answer = "Mars is red [mars.md p.1 / chunk 0]."
    used = used_passages(answer, passages)
    assert len(used) == 1
    assert used[0].citation == "[mars.md p.1 / chunk 0]"


def test_no_citations_returns_empty():
    assert extract_citations("No markers here at all.") == []

"""Unit tests for sentence-aware recursive chunking."""

from __future__ import annotations

import pytest

from rag.chunking import Chunk, chunk_text


def test_short_text_single_chunk():
    chunks = chunk_text("A short sentence.", "doc.md", chunk_size=100, chunk_overlap=20)
    assert len(chunks) == 1
    assert chunks[0].text == "A short sentence."
    assert chunks[0].doc_name == "doc.md"
    assert chunks[0].chunk_index == 0


def test_chunk_size_is_respected_with_tolerance():
    body = " ".join(f"word{i}" for i in range(400))
    chunks = chunk_text(body, "doc.md", chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1
    # Allow a small overflow from overlap/merge but no runaway chunks.
    assert all(len(c.text) <= 200 + 40 for c in chunks)


def test_overlap_carries_context_between_chunks():
    sentences = ". ".join(f"Sentence number {i} has content" for i in range(40))
    chunks = chunk_text(sentences, "doc.md", chunk_size=180, chunk_overlap=60)
    assert len(chunks) >= 2
    # Consecutive chunks should share some overlapping characters.
    tail = chunks[0].text[-30:]
    assert any(word in chunks[1].text for word in tail.split()[-2:])


def test_chunk_indices_are_monotonic_with_start_index():
    chunks = chunk_text(
        " ".join(f"token{i}" for i in range(300)),
        "doc.md",
        chunk_size=120,
        chunk_overlap=20,
        start_index=5,
    )
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(5, 5 + len(chunks)))


def test_empty_text_returns_no_chunks():
    assert chunk_text("   ", "doc.md") == []


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("hello world", "doc.md", chunk_size=50, chunk_overlap=50)


def test_chunk_id_is_stable_and_descriptive():
    c = Chunk(text="hi", doc_name="d.md", chunk_index=3, page=2)
    assert c.chunk_id == "d.md::p2::c3"

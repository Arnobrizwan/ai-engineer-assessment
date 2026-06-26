"""Sentence-aware recursive chunking.

We split documents into overlapping chunks using a hierarchy of separators
(paragraph -> sentence -> word). This keeps semantically coherent units
together while bounding chunk size, and the overlap preserves context across
boundaries so retrieval rarely cuts an answer in half.

Pages are tracked for citation purposes. For plain-text/markdown we treat the
whole document as page 1; for PDFs the ingest layer supplies per-page text and
calls :func:`chunk_text` per page so ``page`` stays accurate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Separators tried in order, from coarsest (paragraph) to finest (character).
_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]


@dataclass
class Chunk:
    """A retrievable unit of text plus its provenance metadata."""

    text: str
    doc_name: str
    chunk_index: int
    page: int = 1
    # Stable id used by the vector store / citation layer.
    chunk_id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.chunk_id:
            self.chunk_id = f"{self.doc_name}::p{self.page}::c{self.chunk_index}"


def _split_recursive(text: str, separators: list[str], chunk_size: int) -> list[str]:
    """Recursively split ``text`` so each piece is <= ``chunk_size`` chars.

    Falls back to the next finer separator whenever a piece is still too large,
    and finally to a hard character slice if even single tokens overflow.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    if not separators:
        # No separators left: hard-slice to guarantee termination.
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep, *rest = separators
    parts = text.split(sep)
    pieces: list[str] = []
    for part in parts:
        candidate = part + sep
        if len(candidate) <= chunk_size:
            pieces.append(candidate)
        else:
            pieces.extend(_split_recursive(part, rest, chunk_size))
    return [p for p in pieces if p.strip()]


def _merge_with_overlap(
    pieces: list[str], chunk_size: int, overlap: int
) -> list[str]:
    """Greedily merge small ``pieces`` up to ``chunk_size`` with char ``overlap``."""
    merged: list[str] = []
    current = ""
    for piece in pieces:
        if len(current) + len(piece) <= chunk_size or not current:
            current += piece
        else:
            merged.append(current.strip())
            # Start next chunk with a tail-overlap of the previous one.
            tail = current[-overlap:] if overlap > 0 else ""
            current = tail + piece
    if current.strip():
        merged.append(current.strip())
    return merged


def chunk_text(
    text: str,
    doc_name: str,
    *,
    page: int = 1,
    chunk_size: int = 700,
    chunk_overlap: int = 120,
    start_index: int = 0,
) -> list[Chunk]:
    """Chunk a single page/document of ``text`` into overlapping :class:`Chunk`s.

    ``start_index`` lets the caller keep ``chunk_index`` globally monotonic when
    chunking a multi-page document page by page.
    """
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    pieces = _split_recursive(text, list(_SEPARATORS), chunk_size)
    merged = _merge_with_overlap(pieces, chunk_size, chunk_overlap)

    return [
        Chunk(text=body, doc_name=doc_name, chunk_index=start_index + i, page=page)
        for i, body in enumerate(merged)
    ]

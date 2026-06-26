"""Citation formatting and source-passage extraction.

A citation has the canonical form ``[doc_name p.X / chunk N]``. The agent is
prompted to cite using exactly this marker, and the UI renders the underlying
passages so a reader can verify every claim against its source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .chunking import Chunk
from .retriever import ScoredChunk

# Matches markers like: [photosynthesis.md p.1 / chunk 3]
_CITATION_RE = re.compile(
    r"\[(?P<doc>[^\]\[]+?)\s+p\.(?P<page>\d+)\s*/\s*chunk\s*(?P<chunk>\d+)\]"
)


def format_citation(chunk: Chunk) -> str:
    """Return the canonical citation marker for a chunk."""
    return f"[{chunk.doc_name} p.{chunk.page} / chunk {chunk.chunk_index}]"


@dataclass
class SourcePassage:
    """A passage offered to the LLM as evidence, with its citation marker."""

    citation: str
    text: str
    chunk: Chunk


def build_context(scored: Sequence[ScoredChunk]) -> tuple[str, list[SourcePassage]]:
    """Build the evidence block for the prompt and the list of source passages.

    Returns:
        ``(context_text, passages)`` where ``context_text`` is the numbered,
        citation-tagged evidence to feed the LLM, and ``passages`` is the
        structured list the UI renders.
    """
    passages: list[SourcePassage] = []
    lines: list[str] = []
    for sc in scored:
        marker = format_citation(sc.chunk)
        passages.append(
            SourcePassage(citation=marker, text=sc.chunk.text, chunk=sc.chunk)
        )
        lines.append(f"{marker}\n{sc.chunk.text}")
    return "\n\n".join(lines), passages


def extract_citations(answer: str) -> list[str]:
    """Return all citation markers found in an answer, in order, de-duplicated."""
    seen: list[str] = []
    for match in _CITATION_RE.finditer(answer):
        marker = (
            f"[{match.group('doc').strip()} p.{match.group('page')} "
            f"/ chunk {match.group('chunk')}]"
        )
        if marker not in seen:
            seen.append(marker)
    return seen


def used_passages(
    answer: str, passages: Sequence[SourcePassage]
) -> list[SourcePassage]:
    """Filter ``passages`` down to those actually cited in ``answer``."""
    cited = set(extract_citations(answer))
    return [p for p in passages if p.citation in cited]

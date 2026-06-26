"""Convenience wiring: documents -> retriever -> agent.

Keeps ``app.py`` and the eval harness from repeating the same assembly. The
embedding function is taken from a live :class:`~rag.llm.LLMClient` unless one
is injected (tests pass a fake to stay offline).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .agent import AgenticRAG, SupportsChat
from .chunking import Chunk
from .config import Config, get_config
from .ingest import ingest_directory, ingest_paths
from .llm import LLMClient
from .retriever import EmbedFn, HybridRetriever


def build_retriever(
    chunks: Sequence[Chunk],
    *,
    embed_fn: EmbedFn | None = None,
    config: Config | None = None,
) -> HybridRetriever:
    """Build a hybrid retriever over ``chunks``.

    If ``embed_fn`` is None the dense index is skipped (sparse-only fallback),
    which keeps things usable even when no embedding server is reachable.
    """
    config = config or get_config()
    return HybridRetriever(chunks, embed_fn=embed_fn, rrf_k=config.rrf_k)


def build_agent(
    chunks: Sequence[Chunk],
    *,
    llm: SupportsChat | None = None,
    embed_fn: EmbedFn | None = None,
    config: Config | None = None,
) -> AgenticRAG:
    """Assemble a ready-to-run :class:`AgenticRAG` over ``chunks``."""
    config = config or get_config()
    client = llm or LLMClient(config)
    if embed_fn is None and isinstance(client, LLMClient):
        embed_fn = client.embed
    retriever = build_retriever(chunks, embed_fn=embed_fn, config=config)
    return AgenticRAG(retriever, client, config)


def build_agent_from_dir(
    directory: Path | None = None,
    *,
    llm: SupportsChat | None = None,
    embed_fn: EmbedFn | None = None,
    config: Config | None = None,
) -> AgenticRAG:
    """Ingest a directory of docs and return an agent over them."""
    config = config or get_config()
    chunks = ingest_directory(directory, config)
    return build_agent(chunks, llm=llm, embed_fn=embed_fn, config=config)


def build_agent_from_paths(
    paths,
    *,
    llm: SupportsChat | None = None,
    embed_fn: EmbedFn | None = None,
    config: Config | None = None,
) -> AgenticRAG:
    """Ingest specific files and return an agent over them."""
    config = config or get_config()
    chunks = ingest_paths(paths, config)
    return build_agent(chunks, llm=llm, embed_fn=embed_fn, config=config)

"""Agentic RAG package.

Public surface:

* :class:`~rag.config.Config` -- configuration loaded from ``.env``.
* :class:`~rag.llm.LLMClient` -- Ollama chat + embedding client.
* :func:`~rag.ingest.ingest_directory` -- load + chunk documents.
* :class:`~rag.retriever.HybridRetriever` -- dense+sparse+RRF+rerank retrieval.
* :class:`~rag.agent.AgenticRAG` -- the reasoning loop.
"""

from .agent import AgenticRAG, AgentResult
from .config import Config, get_config
from .ingest import ingest_directory, ingest_paths
from .llm import LLMClient
from .retriever import HybridRetriever, reciprocal_rank_fusion

__all__ = [
    "AgenticRAG",
    "AgentResult",
    "Config",
    "get_config",
    "ingest_directory",
    "ingest_paths",
    "LLMClient",
    "HybridRetriever",
    "reciprocal_rank_fusion",
]

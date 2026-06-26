"""Central configuration loaded from environment / .env file.

All runtime knobs live here so the rest of the package never reads
``os.environ`` directly. Values are loaded once via ``python-dotenv``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (the folder containing this package's parent).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Config:
    """Immutable application configuration.

    Attributes are populated from environment variables with sensible
    defaults so the app runs out of the box against a local Ollama server.
    """

    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "llama3.1:8b")
    )
    embed_model: str = field(
        default_factory=lambda: os.getenv("EMBED_MODEL", "nomic-embed-text")
    )

    # Retrieval / chunking knobs (overridable via env but rarely changed).
    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "700"))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "120"))
    )
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "5")))
    rrf_k: int = field(default_factory=lambda: int(os.getenv("RRF_K", "60")))
    max_agent_iterations: int = field(
        default_factory=lambda: int(os.getenv("MAX_AGENT_ITERATIONS", "2"))
    )

    # Where the bundled sample documents live.
    data_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "data")

    @property
    def ollama_native_url(self) -> str:
        """Base URL for Ollama's native ``/api/*`` endpoints."""
        return self.ollama_base_url.rstrip("/")


def get_config() -> Config:
    """Return a fresh :class:`Config` (re-reads current environment)."""
    return Config()

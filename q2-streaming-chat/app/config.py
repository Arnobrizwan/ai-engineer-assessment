"""Application configuration loaded from environment variables.

All runtime configuration is read from a ``.env`` file (via ``python-dotenv``)
falling back to process environment variables. See ``.env.example`` for the
full list of supported keys.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load variables from a local .env file if present. This is a no-op when the
# file does not exist, which keeps the import safe for test environments.
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Immutable container for application settings.

    Attributes:
        ollama_base_url: Base URL of the Ollama server (no trailing slash).
        llm_model: Name of the Ollama model used for generation.
        database_url: SQLAlchemy database URL for session/message persistence.
        request_timeout: Per-request HTTP timeout (seconds) for the LLM call.
    """

    ollama_base_url: str
    llm_model: str
    database_url: str
    request_timeout: float


def get_settings() -> Settings:
    """Build a :class:`Settings` instance from the current environment.

    Reading lazily (rather than at import time) lets tests override values via
    ``monkeypatch.setenv`` before the first call.

    Returns:
        A populated, immutable :class:`Settings` object.
    """

    return Settings(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
        llm_model=os.getenv("LLM_MODEL", "llama3.1:8b"),
        database_url=os.getenv("DATABASE_URL", "sqlite:///./chat.db"),
        request_timeout=float(os.getenv("LLM_REQUEST_TIMEOUT", "120")),
    )

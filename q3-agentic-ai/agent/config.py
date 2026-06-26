"""Centralised configuration for the SQL Analytics Agent.

All runtime configuration is sourced from environment variables (optionally
loaded from a local ``.env`` file via ``python-dotenv``). This keeps secrets and
environment-specific values out of the codebase and makes the agent portable
across machines.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a .env file if present. Calling this at import time means
# every module that imports ``config`` gets a consistent view of the
# environment without each having to call ``load_dotenv`` itself.
load_dotenv()

# Project root = directory that contains this ``agent`` package's parent.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


def _sqlite_path_from_url(database_url: str) -> Path:
    """Convert a ``sqlite:///`` SQLAlchemy-style URL into a filesystem path.

    Args:
        database_url: A URL such as ``sqlite:///./analytics.db``.

    Returns:
        An absolute :class:`~pathlib.Path` to the SQLite file. Relative paths
        are resolved against :data:`PROJECT_ROOT` so the database location is
        stable regardless of the current working directory.
    """
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        # Fall back to treating the whole value as a path.
        raw = database_url
    else:
        raw = database_url[len(prefix):]
    path = Path(raw)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


@dataclass(frozen=True)
class Settings:
    """Immutable, typed view of the agent's configuration.

    Attributes:
        ollama_base_url: Base URL of the local Ollama server.
        llm_model: Name of the generation model used for tool calling.
        database_url: SQLAlchemy-style SQLite URL.
        db_path: Resolved absolute path to the SQLite database file.
        max_iterations: Hard cap on agent reasoning/tool-call iterations.
        sql_row_limit: Maximum rows returned by ``run_sql`` (defensive cap).
        request_timeout: Seconds to wait for an Ollama HTTP response.
    """

    ollama_base_url: str
    llm_model: str
    database_url: str
    db_path: Path
    max_iterations: int
    sql_row_limit: int
    request_timeout: float


def get_settings() -> Settings:
    """Build a :class:`Settings` instance from the current environment.

    Returns:
        A fully-populated, immutable :class:`Settings` object.
    """
    database_url = os.getenv("DATABASE_URL", "sqlite:///./analytics.db")
    return Settings(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        llm_model=os.getenv("LLM_MODEL", "llama3.1:8b"),
        database_url=database_url,
        db_path=_sqlite_path_from_url(database_url),
        max_iterations=int(os.getenv("AGENT_MAX_ITERATIONS", "8")),
        sql_row_limit=int(os.getenv("SQL_ROW_LIMIT", "1000")),
        request_timeout=float(os.getenv("OLLAMA_TIMEOUT", "120")),
    )


# A module-level default that most callers can use directly.
settings: Settings = get_settings()

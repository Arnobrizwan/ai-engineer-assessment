"""Shared pytest fixtures.

Provides a freshly seeded temporary SQLite database and a bound
:class:`~agent.tools.Toolbox` so tests never touch the real ``analytics.db`` and
require no live LLM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

from agent.db import seed
from agent.tools import Toolbox


@pytest.fixture()
def seeded_db(tmp_path: Path) -> Path:
    """Create and seed a temporary SQLite database; return its path."""
    db_path = tmp_path / "test_analytics.db"
    seed(db_path)
    return db_path


@pytest.fixture()
def toolbox(seeded_db: Path) -> Toolbox:
    """A :class:`Toolbox` bound to the temporary seeded database."""
    return Toolbox(seeded_db, row_limit=1000)

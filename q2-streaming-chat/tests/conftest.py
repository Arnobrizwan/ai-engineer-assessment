"""Shared pytest fixtures.

All fixtures use a fresh, isolated SQLite database so tests never touch the
real ``chat.db`` and never require a running LLM. The database engine is
rebound to a temporary file per test to guarantee isolation across modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app import db as db_module


@pytest.fixture()
def engine(tmp_path: Path):
    """Create a temporary file-backed SQLite engine with tables created."""

    url = f"sqlite:///{tmp_path / 'test.db'}"
    eng = create_engine(url, connect_args={"check_same_thread": False}, future=True)
    db_module.Base.metadata.create_all(bind=eng)
    yield eng
    eng.dispose()


@pytest.fixture()
def db_session(engine) -> Iterator[Session]:
    """Yield a SQLAlchemy session bound to the temporary engine."""

    TestingSession = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)
    session = TestingSession()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def client(engine, monkeypatch) -> Iterator["TestClient"]:
    """Provide a FastAPI ``TestClient`` wired to the temporary database.

    The app's ``get_db`` dependency is overridden so requests use the isolated
    test engine instead of the production database.
    """

    from fastapi.testclient import TestClient

    from app.main import app

    TestingSession = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)

    def override_get_db() -> Iterator[Session]:
        session = TestingSession()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[db_module.get_db] = override_get_db
    # Prevent startup hook from creating tables on the real engine.
    monkeypatch.setattr(db_module, "init_db", lambda: None)

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()

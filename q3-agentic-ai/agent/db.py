"""Database creation, seeding, and connection helpers.

The agent operates against a small but realistic e-commerce analytics schema:

    customers (1) ──< orders (1) ──< order_items >── (1) products

Seeding is **idempotent**: running it repeatedly drops and recreates the demo
tables so the dataset is deterministic for both the app and the test-suite.
"""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Sequence

# --- Schema definition -------------------------------------------------------

SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE customers (
        id            INTEGER PRIMARY KEY,
        name          TEXT    NOT NULL,
        email         TEXT    NOT NULL UNIQUE,
        country       TEXT    NOT NULL,
        signup_date   TEXT    NOT NULL          -- ISO 8601 date
    )
    """,
    """
    CREATE TABLE products (
        id            INTEGER PRIMARY KEY,
        name          TEXT    NOT NULL,
        category      TEXT    NOT NULL,
        price         REAL    NOT NULL CHECK (price >= 0)
    )
    """,
    """
    CREATE TABLE orders (
        id            INTEGER PRIMARY KEY,
        customer_id   INTEGER NOT NULL REFERENCES customers(id),
        order_date    TEXT    NOT NULL,         -- ISO 8601 date
        status        TEXT    NOT NULL          -- e.g. completed, cancelled
    )
    """,
    """
    CREATE TABLE order_items (
        id            INTEGER PRIMARY KEY,
        order_id      INTEGER NOT NULL REFERENCES orders(id),
        product_id    INTEGER NOT NULL REFERENCES products(id),
        quantity      INTEGER NOT NULL CHECK (quantity > 0),
        unit_price    REAL    NOT NULL CHECK (unit_price >= 0)
    )
    """,
)

TABLE_NAMES: tuple[str, ...] = ("customers", "products", "orders", "order_items")


# --- Connection helpers ------------------------------------------------------


def connect(db_path: Path | str) -> sqlite3.Connection:
    """Open a SQLite connection with sensible defaults.

    Row results are returned as :class:`sqlite3.Row` objects so callers can use
    column-name access. Foreign-key enforcement is enabled.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        An open :class:`sqlite3.Connection`.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# --- Seed data ---------------------------------------------------------------

_BASE_DATE = date(2024, 1, 1)


def _customers() -> list[tuple[int, str, str, str, str]]:
    rows = [
        (1, "Alice Johnson", "alice@example.com", "USA", "2024-01-05"),
        (2, "Bob Smith", "bob@example.com", "USA", "2024-01-20"),
        (3, "Carlos Diaz", "carlos@example.com", "Mexico", "2024-02-02"),
        (4, "Diana Prince", "diana@example.com", "UK", "2024-02-15"),
        (5, "Erik Larsson", "erik@example.com", "Sweden", "2024-03-01"),
        (6, "Fatima Noor", "fatima@example.com", "UAE", "2024-03-18"),
        (7, "Grace Lee", "grace@example.com", "Singapore", "2024-04-04"),
        (8, "Hiroshi Tanaka", "hiroshi@example.com", "Japan", "2024-04-22"),
    ]
    return rows


def _products() -> list[tuple[int, str, str, float]]:
    return [
        (1, "Wireless Mouse", "Accessories", 25.0),
        (2, "Mechanical Keyboard", "Accessories", 80.0),
        (3, "27-inch Monitor", "Displays", 320.0),
        (4, "USB-C Hub", "Accessories", 45.0),
        (5, "Laptop Stand", "Furniture", 60.0),
        (6, "Noise-Cancelling Headphones", "Audio", 200.0),
        (7, "Webcam 1080p", "Accessories", 70.0),
        (8, "Ergonomic Chair", "Furniture", 450.0),
    ]


def _orders() -> list[tuple[int, int, str, str]]:
    # (id, customer_id, order_date, status)
    specs = [
        (1, 1, 4, "completed"),
        (2, 1, 40, "completed"),
        (3, 2, 12, "completed"),
        (4, 3, 33, "cancelled"),
        (5, 3, 60, "completed"),
        (6, 4, 50, "completed"),
        (7, 5, 70, "completed"),
        (8, 6, 80, "completed"),
        (9, 7, 95, "completed"),
        (10, 8, 110, "completed"),
        (11, 2, 120, "completed"),
        (12, 4, 130, "completed"),
    ]
    return [
        (oid, cid, (_BASE_DATE + timedelta(days=offset)).isoformat(), status)
        for (oid, cid, offset, status) in specs
    ]


def _order_items() -> list[tuple[int, int, int, int, float]]:
    # (id, order_id, product_id, quantity, unit_price)
    prices = {pid: price for (pid, _n, _c, price) in _products()}
    raw = [
        (1, 1, 1, 2),
        (2, 1, 2, 1),
        (3, 2, 3, 1),
        (4, 3, 6, 1),
        (5, 3, 4, 2),
        (6, 4, 8, 1),
        (7, 5, 2, 1),
        (8, 5, 7, 1),
        (9, 6, 3, 2),
        (10, 7, 6, 1),
        (11, 7, 1, 3),
        (12, 8, 8, 1),
        (13, 9, 5, 2),
        (14, 10, 3, 1),
        (15, 11, 2, 2),
        (16, 11, 4, 1),
        (17, 12, 6, 1),
        (18, 12, 7, 2),
    ]
    return [(iid, oid, pid, qty, prices[pid]) for (iid, oid, pid, qty) in raw]


def _insert_many(
    conn: sqlite3.Connection, sql: str, rows: Iterable[Sequence[object]]
) -> None:
    conn.executemany(sql, list(rows))


def seed(db_path: Path | str) -> Path:
    """Create the schema and populate it with deterministic demo data.

    The operation is idempotent: existing demo tables are dropped first, so the
    resulting database always contains exactly the seed dataset.

    Args:
        db_path: Path to the SQLite database file to (re)create.

    Returns:
        The resolved path to the seeded database.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = connect(path)
    try:
        # Drop in reverse dependency order to satisfy foreign keys.
        for table in reversed(TABLE_NAMES):
            conn.execute(f"DROP TABLE IF EXISTS {table}")
        for statement in SCHEMA_STATEMENTS:
            conn.execute(statement)

        _insert_many(
            conn,
            "INSERT INTO customers (id, name, email, country, signup_date) "
            "VALUES (?, ?, ?, ?, ?)",
            _customers(),
        )
        _insert_many(
            conn,
            "INSERT INTO products (id, name, category, price) VALUES (?, ?, ?, ?)",
            _products(),
        )
        _insert_many(
            conn,
            "INSERT INTO orders (id, customer_id, order_date, status) "
            "VALUES (?, ?, ?, ?)",
            _orders(),
        )
        _insert_many(
            conn,
            "INSERT INTO order_items (id, order_id, product_id, quantity, unit_price) "
            "VALUES (?, ?, ?, ?, ?)",
            _order_items(),
        )
        conn.commit()
    finally:
        conn.close()
    return path.resolve()


def ensure_seeded(db_path: Path | str) -> Path:
    """Seed the database only if it does not already contain the demo tables.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        The resolved path to the database.
    """
    path = Path(db_path)
    if not path.exists():
        return seed(path)
    conn = connect(path)
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='customers'"
        )
        if cur.fetchone() is None:
            conn.close()
            return seed(path)
    finally:
        conn.close()
    return path.resolve()

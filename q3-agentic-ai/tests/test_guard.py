"""Tests for the read-only SQL safety guard."""

from __future__ import annotations

import pytest

from agent.tools import UnsafeSQLError, assert_safe_select, is_safe_select

SAFE_QUERIES = [
    "SELECT * FROM customers",
    "select id, name from customers where country = 'USA'",
    "  SELECT COUNT(*) FROM orders  ",
    "WITH cte AS (SELECT id FROM orders) SELECT * FROM cte",
    "SELECT * FROM products; ",  # single statement with trailing semicolon
    "SELECT name FROM products -- a comment\n WHERE price > 10",
]

UNSAFE_QUERIES = [
    "",
    "   ",
    "DELETE FROM customers",
    "UPDATE customers SET name = 'x'",
    "DROP TABLE customers",
    "INSERT INTO customers (id) VALUES (99)",
    "ALTER TABLE customers ADD COLUMN x TEXT",
    "CREATE TABLE evil (id INT)",
    "TRUNCATE TABLE orders",
    "PRAGMA table_info(customers)",
    "ATTACH DATABASE 'x.db' AS y",
    "SELECT 1; DROP TABLE customers",  # stacked statements
    "SELECT 1; SELECT 2",  # multiple selects
    "SELECT * FROM customers; DELETE FROM orders;",
    "/* hide */ DROP TABLE customers",
    "SELECT * FROM t WHERE x = 1; UPDATE t SET x = 2",
    "EXPLAIN DELETE FROM customers",  # contains DELETE keyword
]


@pytest.mark.parametrize("query", SAFE_QUERIES)
def test_safe_queries_allowed(query: str) -> None:
    assert is_safe_select(query) is True
    # Should not raise.
    assert_safe_select(query)


@pytest.mark.parametrize("query", UNSAFE_QUERIES)
def test_unsafe_queries_rejected(query: str) -> None:
    assert is_safe_select(query) is False
    with pytest.raises(UnsafeSQLError):
        assert_safe_select(query)


def test_comment_cannot_smuggle_keyword() -> None:
    # A DROP hidden only in a comment is stripped, leaving a safe SELECT.
    assert is_safe_select("SELECT 1 -- DROP TABLE customers") is True
    # But a real DROP after a comment is still caught.
    assert is_safe_select("-- ok\nDROP TABLE customers") is False

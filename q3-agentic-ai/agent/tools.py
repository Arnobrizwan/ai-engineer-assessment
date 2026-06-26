"""Agent tools: schema introspection, guarded SQL execution, and charting.

Each tool is a plain Python function paired with an OpenAI/Ollama-compatible
JSON schema (see :data:`TOOL_SCHEMAS`). The agent loop selects a tool by name,
the function executes against the real SQLite database, and the structured
result is fed back to the model as an observation.

The most important piece here is :func:`is_safe_select` / :func:`run_sql`, which
implement the **read-only guardrail**: only a single ``SELECT`` (or ``WITH ...
SELECT``) statement is ever allowed to reach the database.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import db as db_module

# --- SQL safety guard --------------------------------------------------------

# Keywords that indicate data/schema mutation or otherwise disallowed verbs.
# Matched as whole words, case-insensitively.
_FORBIDDEN_KEYWORDS: frozenset[str] = frozenset(
    {
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "ALTER",
        "CREATE",
        "REPLACE",
        "TRUNCATE",
        "ATTACH",
        "DETACH",
        "PRAGMA",
        "VACUUM",
        "REINDEX",
        "GRANT",
        "REVOKE",
        "MERGE",
        "EXEC",
        "EXECUTE",
    }
)

_COMMENT_RE = re.compile(r"(--[^\n]*)|(/\*.*?\*/)", re.DOTALL)
_WORD_RE = re.compile(r"[A-Za-z_]+")


class UnsafeSQLError(ValueError):
    """Raised when a SQL string fails the read-only safety guard."""


def _strip_comments(sql: str) -> str:
    """Remove SQL line and block comments so they cannot smuggle keywords."""
    return _COMMENT_RE.sub(" ", sql)


def _split_statements(sql: str) -> list[str]:
    """Split on semicolons, ignoring a single trailing terminator.

    A naive split is sufficient here because the guard rejects any string that
    yields more than one **non-empty** statement, which is exactly the
    multi-statement / stacked-query attack we want to block.
    """
    parts = [part.strip() for part in sql.split(";")]
    return [part for part in parts if part]


def is_safe_select(sql: str) -> bool:
    """Return ``True`` only if ``sql`` is a single read-only SELECT statement.

    The guard enforces, in order:

    1. Non-empty input.
    2. Exactly one statement (blocks stacked/multi-statement injection).
    3. The statement starts with ``SELECT`` or ``WITH`` (CTEs ending in SELECT).
    4. No forbidden DML/DDL/administrative keyword appears anywhere (checked on
       comment-stripped text, as whole words).

    Args:
        sql: The candidate SQL string.

    Returns:
        ``True`` if the query is a safe read-only SELECT, ``False`` otherwise.
    """
    if not sql or not sql.strip():
        return False

    cleaned = _strip_comments(sql)
    statements = _split_statements(cleaned)
    if len(statements) != 1:
        return False

    statement = statements[0]
    first_word_match = _WORD_RE.search(statement)
    if first_word_match is None:
        return False

    first_word = first_word_match.group(0).upper()
    if first_word not in {"SELECT", "WITH"}:
        return False

    tokens = {tok.upper() for tok in _WORD_RE.findall(statement)}
    if tokens & _FORBIDDEN_KEYWORDS:
        return False

    return True


def assert_safe_select(sql: str) -> None:
    """Raise :class:`UnsafeSQLError` if ``sql`` is not a safe SELECT.

    Args:
        sql: The candidate SQL string.

    Raises:
        UnsafeSQLError: If the query is rejected by :func:`is_safe_select`.
    """
    if not is_safe_select(sql):
        raise UnsafeSQLError(
            "Rejected: only a single read-only SELECT statement is permitted. "
            "DML/DDL (INSERT, UPDATE, DELETE, DROP, ...), multiple statements, "
            "and PRAGMA/ATTACH are not allowed."
        )


# --- Tool result container ---------------------------------------------------


@dataclass
class SQLResult:
    """Structured result of a successful ``run_sql`` call.

    Attributes:
        columns: Ordered list of column names.
        rows: List of row tuples (native Python types).
        row_count: Number of rows returned.
        truncated: Whether the result was capped at the configured row limit.
    """

    columns: list[str]
    rows: list[tuple[Any, ...]]
    row_count: int
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation for the model observation."""
        return {
            "columns": self.columns,
            "rows": [list(r) for r in self.rows],
            "row_count": self.row_count,
            "truncated": self.truncated,
        }


# --- Toolbox -----------------------------------------------------------------


class Toolbox:
    """Bind tool functions to a concrete SQLite database path.

    A :class:`Toolbox` is the executable side of the agent's tools. The agent
    loop looks up a tool by name in :attr:`dispatch` and calls it with the
    model-supplied arguments.

    Args:
        db_path: Path to the SQLite database the tools operate on.
        row_limit: Maximum number of rows ``run_sql`` will return.
    """

    def __init__(self, db_path: Path | str, row_limit: int = 1000) -> None:
        self.db_path = Path(db_path)
        self.row_limit = int(row_limit)

    # -- schema introspection tools --

    def list_tables(self) -> dict[str, Any]:
        """List user tables available in the database.

        Returns:
            A dict ``{"tables": [...]}`` with the table names.
        """
        conn = db_module.connect(self.db_path)
        try:
            cur = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
            names = [row[0] for row in cur.fetchall()]
        finally:
            conn.close()
        return {"tables": names}

    def get_schema(self, table: str) -> dict[str, Any]:
        """Return the column schema for ``table``.

        Args:
            table: Table name to introspect.

        Returns:
            A dict with the table name and a list of column descriptors
            (name, type, not-null flag, primary-key flag), or an ``error`` key
            if the table does not exist.
        """
        # Validate the identifier to keep PRAGMA injection-free. SQLite table
        # names here are simple identifiers.
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table or ""):
            return {"error": f"Invalid table name: {table!r}"}

        conn = db_module.connect(self.db_path)
        try:
            cur = conn.execute(f"PRAGMA table_info({table})")
            info = cur.fetchall()
            if not info:
                return {"error": f"Table not found: {table!r}"}
            columns = [
                {
                    "name": row["name"],
                    "type": row["type"],
                    "not_null": bool(row["notnull"]),
                    "primary_key": bool(row["pk"]),
                }
                for row in info
            ]
        finally:
            conn.close()
        return {"table": table, "columns": columns}

    # -- guarded query tool --

    def run_sql(self, query: str) -> dict[str, Any]:
        """Execute a read-only SELECT and return rows, or a structured error.

        The query is first checked by the safety guard. Execution errors
        (bad column, syntax, etc.) are caught and returned as an ``error`` field
        so the agent can read the message and self-correct on its next turn.

        Args:
            query: A single read-only SELECT statement.

        Returns:
            On success, the :meth:`SQLResult.to_dict` payload. On failure, a
            dict with an ``error`` string.
        """
        try:
            assert_safe_select(query)
        except UnsafeSQLError as exc:
            return {"error": str(exc)}

        conn = db_module.connect(self.db_path)
        try:
            cur = conn.execute(query)
            columns = [d[0] for d in cur.description] if cur.description else []
            fetched = cur.fetchmany(self.row_limit + 1)
            truncated = len(fetched) > self.row_limit
            rows = [tuple(r) for r in fetched[: self.row_limit]]
        except sqlite3.Error as exc:
            return {"error": f"SQL execution error: {exc}"}
        finally:
            conn.close()

        result = SQLResult(
            columns=columns,
            rows=rows,
            row_count=len(rows),
            truncated=truncated,
        )
        return result.to_dict()

    # -- charting tool --

    def make_chart(
        self,
        labels: list[Any],
        values: list[float],
        chart_type: str = "bar",
        title: str = "",
    ) -> dict[str, Any]:
        """Describe a simple chart from a label/value series.

        This tool is intentionally rendering-agnostic: it validates and returns
        a chart *specification* (the Streamlit UI renders it). Keeping the tool
        pure makes it trivial to unit-test without a display backend.

        Args:
            labels: Category labels for the x-axis.
            values: Numeric values aligned with ``labels``.
            chart_type: One of ``"bar"`` or ``"line"``.
            title: Optional chart title.

        Returns:
            A chart specification dict, or an ``error`` field on invalid input.
        """
        if chart_type not in {"bar", "line"}:
            return {"error": f"Unsupported chart_type: {chart_type!r}"}
        if len(labels) != len(values):
            return {"error": "labels and values must have the same length"}
        try:
            numeric = [float(v) for v in values]
        except (TypeError, ValueError):
            return {"error": "values must be numeric"}
        return {
            "chart_type": chart_type,
            "title": title,
            "labels": [str(label) for label in labels],
            "values": numeric,
        }

    # -- dispatch table --

    @property
    def dispatch(self) -> dict[str, Callable[..., dict[str, Any]]]:
        """Map tool names to their bound callables."""
        return {
            "list_tables": lambda: self.list_tables(),
            "get_schema": self.get_schema,
            "run_sql": self.run_sql,
            "make_chart": self.make_chart,
        }

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute the named tool with keyword ``arguments``.

        Args:
            name: Tool name (must exist in :attr:`dispatch`).
            arguments: Keyword arguments supplied by the model.

        Returns:
            The tool's structured result, or an ``error`` dict for an unknown
            tool or bad argument shape.
        """
        fn = self.dispatch.get(name)
        if fn is None:
            return {"error": f"Unknown tool: {name!r}"}
        try:
            return fn(**(arguments or {}))
        except TypeError as exc:
            return {"error": f"Bad arguments for tool {name!r}: {exc}"}


# --- JSON tool schemas (Ollama / OpenAI function-calling format) -------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_tables",
            "description": "List all tables available in the analytics database.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_schema",
            "description": (
                "Get the column schema (names, types, keys) for a single table. "
                "Call this before writing SQL so column names are correct."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {
                        "type": "string",
                        "description": "The exact table name to introspect.",
                    }
                },
                "required": ["table"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": (
                "Execute a single READ-ONLY SQL SELECT statement against the "
                "SQLite database and return the result rows. Only SELECT (or "
                "WITH ... SELECT) is allowed; any DML/DDL or multiple statements "
                "are rejected. If a query errors, read the error and try again."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A single read-only SELECT statement.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "make_chart",
            "description": (
                "Build a simple bar or line chart specification from a series of "
                "labels and numeric values (e.g. to visualise a SQL result)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Category labels for the x-axis.",
                    },
                    "values": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Numeric values aligned with labels.",
                    },
                    "chart_type": {
                        "type": "string",
                        "enum": ["bar", "line"],
                        "description": "Chart type to render.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Optional chart title.",
                    },
                },
                "required": ["labels", "values"],
            },
        },
    },
]


def tool_names() -> list[str]:
    """Return the list of tool names exposed to the model."""
    return [schema["function"]["name"] for schema in TOOL_SCHEMAS]

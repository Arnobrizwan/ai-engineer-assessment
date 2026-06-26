"""Tests for the schema, query, and chart tools against a seeded DB."""

from __future__ import annotations

from agent.tools import Toolbox


def test_list_tables(toolbox: Toolbox) -> None:
    tables = toolbox.list_tables()["tables"]
    assert set(tables) == {"customers", "products", "orders", "order_items"}


def test_get_schema_known_table(toolbox: Toolbox) -> None:
    schema = toolbox.get_schema("customers")
    assert schema["table"] == "customers"
    names = [c["name"] for c in schema["columns"]]
    assert "email" in names and "country" in names
    pk = [c["name"] for c in schema["columns"] if c["primary_key"]]
    assert pk == ["id"]


def test_get_schema_unknown_table(toolbox: Toolbox) -> None:
    assert "error" in toolbox.get_schema("does_not_exist")


def test_get_schema_rejects_bad_identifier(toolbox: Toolbox) -> None:
    assert "error" in toolbox.get_schema("customers; DROP TABLE customers")


def test_run_sql_select(toolbox: Toolbox) -> None:
    result = toolbox.run_sql("SELECT COUNT(*) AS n FROM customers")
    assert "error" not in result
    assert result["columns"] == ["n"]
    assert result["rows"][0][0] == 8


def test_run_sql_join(toolbox: Toolbox) -> None:
    result = toolbox.run_sql(
        "SELECT p.name, SUM(oi.quantity * oi.unit_price) AS revenue "
        "FROM order_items oi JOIN products p ON p.id = oi.product_id "
        "GROUP BY p.name ORDER BY revenue DESC"
    )
    assert "error" not in result
    assert result["row_count"] > 0


def test_run_sql_rejects_dml(toolbox: Toolbox) -> None:
    result = toolbox.run_sql("DELETE FROM customers")
    assert "error" in result
    assert "SELECT" in result["error"]


def test_run_sql_reports_bad_query(toolbox: Toolbox) -> None:
    result = toolbox.run_sql("SELECT nope FROM customers")
    assert "error" in result
    assert "SQL execution error" in result["error"]


def test_make_chart_valid(toolbox: Toolbox) -> None:
    chart = toolbox.make_chart(labels=["a", "b"], values=[1, 2], chart_type="bar")
    assert chart["chart_type"] == "bar"
    assert chart["values"] == [1.0, 2.0]


def test_make_chart_mismatched_lengths(toolbox: Toolbox) -> None:
    chart = toolbox.make_chart(labels=["a"], values=[1, 2])
    assert "error" in chart


def test_call_unknown_tool(toolbox: Toolbox) -> None:
    assert "error" in toolbox.call("nope", {})


def test_call_bad_arguments(toolbox: Toolbox) -> None:
    # get_schema requires 'table'; supplying an unexpected kwarg is a TypeError.
    assert "error" in toolbox.call("get_schema", {"wrong": "x"})


def test_call_normalises_run_sql_alias(toolbox: Toolbox) -> None:
    # Small models often emit `sql=`/`statement=` instead of `query=`.
    for alias in ("sql", "statement", "q"):
        result = toolbox.call("run_sql", {alias: "SELECT id FROM customers"})
        assert "error" not in result, f"alias {alias!r} should normalise to 'query'"
        assert result["row_count"] >= 1


def test_call_normalises_get_schema_alias(toolbox: Toolbox) -> None:
    for alias in ("table_name", "name", "tablename"):
        result = toolbox.call("get_schema", {alias: "customers"})
        assert "error" not in result, f"alias {alias!r} should normalise to 'table'"
        assert result["table"] == "customers"


def test_call_canonical_arg_wins_over_alias(toolbox: Toolbox) -> None:
    # If both canonical and alias are present, the canonical value is used.
    result = toolbox.call(
        "run_sql", {"query": "SELECT id FROM customers", "sql": "DROP TABLE customers"}
    )
    assert "error" not in result
    assert result["row_count"] >= 1


def test_row_limit_truncation(seeded_db) -> None:
    small = Toolbox(seeded_db, row_limit=2)
    result = small.run_sql("SELECT id FROM customers ORDER BY id")
    assert result["row_count"] == 2
    assert result["truncated"] is True

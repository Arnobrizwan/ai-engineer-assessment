# Requirement → File/Function Mapping

This checklist maps every assignment requirement for **Question 3 (Agentic AI —
SQL Analytics Agent)** to the exact file and function that satisfies it.

## 1. Agent interacts with an external environment

| Requirement | File / Function | Status |
|---|---|---|
| External environment = local SQLite analytics DB | `agent/db.py` (`connect`, `seed`) | ✅ |
| Realistic e-commerce schema (customers, products, orders, order_items) | `agent/db.py` (`SCHEMA_STATEMENTS`, `_customers`, `_products`, `_orders`, `_order_items`) | ✅ |
| Idempotent seeding with a realistic small dataset | `agent/db.py::seed`, entry point `seed.py` | ✅ |
| Autonomously answers NL questions by planning + tools + synthesis | `agent/loop.py::Agent.run` | ✅ |

## 2. Real tool-calling agent loop (ReAct / function-calling)

| Requirement | File / Function | Status |
|---|---|---|
| `list_tables()` tool | `agent/tools.py::Toolbox.list_tables` + schema in `TOOL_SCHEMAS` | ✅ |
| `get_schema(table)` tool | `agent/tools.py::Toolbox.get_schema` + schema | ✅ |
| `run_sql(query)` read-only SELECT execution | `agent/tools.py::Toolbox.run_sql` | ✅ |
| Guard: reject non-SELECT / multi-statement / DML / DDL | `agent/tools.py::is_safe_select`, `assert_safe_select`, `_FORBIDDEN_KEYWORDS` | ✅ |
| `make_chart(...)` optional tool | `agent/tools.py::Toolbox.make_chart` + schema | ✅ |
| Final `answer` synthesis step | `agent/loop.py::Agent.run` (terminates when model emits no tool calls → `StepType.FINAL`) | ✅ |
| Loop: propose → execute → feed observation → repeat | `agent/loop.py::Agent.run` (main `for` loop, `tool` messages) | ✅ |
| Cap iterations | `agent/loop.py::Agent.run` (`max_iterations`), `agent/config.py` (`max_iterations`) | ✅ |
| Handle SQL errors by feeding error back (self-correction) | `agent/tools.py::Toolbox.run_sql` (returns `{"error": ...}`) + `agent/loop.py` (appends as observation) | ✅ |
| JSON schema per tool | `agent/tools.py::TOOL_SCHEMAS` | ✅ |
| Ollama tool-calling client | `agent/llm.py::OllamaClient.chat` | ✅ |

## 3. Working prototype — Streamlit

| Requirement | File / Function | Status |
|---|---|---|
| User types a business question | `app.py::main` (`st.text_input`) | ✅ |
| Step-by-step trace (thought → tool call → observation) | `app.py::_render_trace`, backed by `agent/loop.py::TraceStep` | ✅ |
| Generated SQL shown | `app.py::main` (`st.code(..., language="sql")`, `result.last_sql`) | ✅ |
| Result table shown | `app.py::_render_observation` (`st.dataframe`) | ✅ |
| Optional chart shown | `app.py::_render_chart`, `result.last_chart` | ✅ |
| Final natural-language answer | `app.py::main` (`result.answer`) | ✅ |

## 4. Quality / test cases

| Requirement | File / Function | Status |
|---|---|---|
| SQL guard rejects DELETE/UPDATE/DROP/multi-statement, allows SELECT | `tests/test_guard.py` | ✅ |
| Schema tools tested | `tests/test_tools.py::test_list_tables`, `test_get_schema_*` | ✅ |
| Agent loop control flow with mocked LLM (right tool + terminates) | `tests/test_loop.py::test_happy_path_calls_tool_then_answers`, `test_schema_then_query_flow`, `test_iteration_cap_enforced` | ✅ |
| Self-correction on a bad query (mock: bad SQL → good SQL) | `tests/test_loop.py::test_self_correction_on_bad_sql` | ✅ |
| Temp seeded SQLite DB in tests | `tests/conftest.py::seeded_db`, `toolbox` | ✅ |
| Tests run WITHOUT a live LLM (mock/monkeypatch) | `tests/test_loop.py::ScriptedLLM` (fake `LLMClient`) | ✅ |
| Optional `@pytest.mark.integration` auto-skipped if Ollama down | `tests/test_integration.py`, `pytest.ini` | ✅ |

## Deliverables

| Deliverable | Location | Status |
|---|---|---|
| `agent/tools.py` (tool fns + JSON schemas + SQL guard) | `agent/tools.py` | ✅ |
| `agent/loop.py` (agentic loop) | `agent/loop.py` | ✅ |
| `agent/llm.py` (Ollama client w/ tool calling) | `agent/llm.py` | ✅ |
| `agent/db.py` (schema + seed) | `agent/db.py` | ✅ |
| `agent/config.py` | `agent/config.py` | ✅ |
| `seed.py` (idempotent DB build) | `seed.py` | ✅ |
| `app.py` (Streamlit) | `app.py` | ✅ |
| `requirements.txt` (pinned) | `requirements.txt` | ✅ |
| `.env.example` | `.env.example` | ✅ |
| `tests/` | `tests/` | ✅ |
| `README.md` (architecture, Investigation: Agentic AI, guardrails, run, testing, limitations) | `README.md` | ✅ |
| `REQUIREMENTS.md` | this file | ✅ |

## Environment constraints

| Constraint | Where satisfied | Status |
|---|---|---|
| Local LLM via Ollama, `llama3.1:8b` | `agent/llm.py`, `.env.example` | ✅ |
| Config via `.env` + `python-dotenv`, `.env.example` provided | `agent/config.py`, `.env.example` | ✅ |
| Keys: `OLLAMA_BASE_URL`, `LLM_MODEL`, `DATABASE_URL` | `.env.example`, `agent/config.py` | ✅ |
| Python 3.11, pinned `requirements.txt` | `requirements.txt` | ✅ |
| Tests use pytest, run without live LLM | `tests/`, `pytest.ini` | ✅ |

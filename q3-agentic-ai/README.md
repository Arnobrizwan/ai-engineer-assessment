# SQL Analytics Agent (Q3 — Agentic AI)

An **advanced agentic AI** that acts as an autonomous data/financial analyst. A
user asks a business question in plain English; the agent **plans**, calls
**tools** against a live SQLite analytics database (its external environment),
**observes** the results, **self-corrects** on errors, and synthesises a
grounded natural-language answer — all driven by a local LLM (`llama3.1:8b` via
Ollama) using real **function/tool calling**.

This is a genuine ReAct-style tool-calling loop, not a prompt-and-pray wrapper:
the model proposes a tool call, the loop executes it against the real database,
feeds the observation back, and repeats until the model produces a final answer
or hits an iteration cap.

---

## Architecture: the agent loop

```
                ┌─────────────────────────────────────────────────────────┐
                │                     STREAMLIT UI (app.py)                 │
                │   question ▸ trace ▸ generated SQL ▸ table ▸ chart ▸ ans  │
                └───────────────────────────┬─────────────────────────────┘
                                            │ question
                                            ▼
   ┌──────────────────────────── Agent.run()  (agent/loop.py) ────────────────────────────┐
   │                                                                                        │
   │   messages = [system prompt, user question]                                           │
   │                                                                                        │
   │   repeat (capped at MAX_ITERATIONS):                                                   │
   │                                                                                        │
   │        ┌────────────┐   tools + messages    ┌──────────────────────────┐              │
   │        │   LLM      │ ◀──────────────────────│  OllamaClient (llm.py)   │              │
   │        │  "brain"   │ ─────────────────────▶ │  /api/chat tool calling  │              │
   │        └─────┬──────┘   tool_calls / answer  └──────────────────────────┘              │
   │              │                                                                          │
   │     has tool calls? ──── no ──▶  FINAL ANSWER  ──▶ return AgentResult                  │
   │              │ yes                                                                      │
   │              ▼                                                                          │
   │     ┌──────────────────────── Toolbox (agent/tools.py) ─────────────────────┐          │
   │     │  list_tables · get_schema(table) · run_sql(query) · make_chart(...)    │          │
   │     │                         └─ SQL SAFETY GUARD ─┘                          │          │
   │     └───────────────────────────────┬────────────────────────────────────────┘          │
   │                                     │ execute against environment                       │
   │                                     ▼                                                    │
   │                         ┌────────────────────────┐                                       │
   │                         │  SQLite analytics.db    │  (agent/db.py — seeded e-commerce)   │
   │                         └────────────────────────┘                                       │
   │                                     │ observation (rows / schema / error)                │
   │                                     ▼                                                    │
   │        append observation as a `tool` message ──▶ loop again (model re-plans)           │
   └────────────────────────────────────────────────────────────────────────────────────────┘
```

The seeded schema is a small but realistic e-commerce dataset:

```
customers (1) ──< orders (1) ──< order_items >── (1) products
```

---

## Investigation: Agentic AI

This project is a concrete study of what makes a system *agentic*. The mapping
below ties each concept to where it lives in the code.

### Core components

| Component | What it is here | Where |
|---|---|---|
| **LLM / "brain"** | `llama3.1:8b` served locally by Ollama, used purely as a reasoning + tool-selection engine. It never touches the database directly — it only *decides* what to do next. | `agent/llm.py` (`OllamaClient`) |
| **Tools** | Typed Python functions, each with an OpenAI/Ollama-compatible JSON schema, that let the agent *act on the world*: `list_tables`, `get_schema`, `run_sql`, `make_chart`. | `agent/tools.py` (`Toolbox`, `TOOL_SCHEMAS`) |
| **Memory** | The running `messages` list is the agent's **working memory** — every thought, tool call, and observation accumulates so each new decision is conditioned on the full interaction so far. This is what enables self-correction. | `agent/loop.py` (`messages`) |
| **Planning** | The system prompt encodes an explicit plan (introspect schema → write SQL → verify → answer); the model then plans *dynamically* turn-by-turn, choosing the next tool based on observations rather than a fixed script. | `agent/loop.py` (`SYSTEM_PROMPT`) |
| **Orchestration loop** | The `Agent.run` control loop that wires brain ↔ tools ↔ environment, enforces the iteration cap, records the trace, and decides termination. | `agent/loop.py` (`Agent.run`) |

### Key characteristics

- **Autonomy** — given only a question, the agent decides *on its own* which
  tables to inspect, what SQL to write, and when it has enough to answer. No
  human picks the tools or the queries.
- **Reactivity** — each decision reacts to the latest observation. If a query
  returns zero rows or an error, the next turn is shaped by that feedback.
- **Tool use** — the agent's only way to affect or read the world is through the
  declared tools; the LLM emits structured tool calls that the loop executes.
- **Goal-directedness** — the loop persists toward a single goal (a grounded
  answer) across multiple steps, rather than producing a one-shot reply.
- **Self-correction** — when `run_sql` returns a structured error, that error is
  fed back as an observation so the model can repair its query. This is exercised
  directly in `tests/test_loop.py::test_self_correction_on_bad_sql` (bad SQL →
  error observation → corrected SQL → answer).

---

## Safety / guardrail design for `run_sql`

`run_sql` is the only tool that executes model-authored SQL, so it is the
primary attack surface. The guard (`agent/tools.py::is_safe_select` /
`assert_safe_select`) enforces **read-only, single-statement SELECTs** with a
defence-in-depth approach:

1. **Comment stripping** — line (`--`) and block (`/* */`) comments are removed
   first so a forbidden verb cannot be smuggled past the parser inside a comment,
   and a real verb cannot be hidden *behind* a comment.
2. **Single-statement enforcement** — the query is split on `;`; anything that
   yields more than one non-empty statement is rejected. This blocks
   **stacked-query injection** like `SELECT 1; DROP TABLE customers`.
3. **Leading-verb check** — the statement must begin with `SELECT` or `WITH`
   (to allow read-only CTEs). `EXPLAIN`, `PRAGMA`, etc. are rejected.
4. **Keyword denylist** — a whole-word, case-insensitive scan rejects any of
   `INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, REPLACE, TRUNCATE, ATTACH,
   DETACH, PRAGMA, VACUUM, REINDEX, GRANT, REVOKE, MERGE, EXEC, EXECUTE`.
5. **Defensive row cap** — even a valid SELECT is capped at `SQL_ROW_LIMIT` rows
   (fetched as limit + 1 to flag truncation) to bound memory and response size.
6. **Identifier validation** — `get_schema` validates the table name against a
   strict identifier regex before interpolating it into a `PRAGMA`, so the schema
   tool cannot be abused for injection either.

Errors (guard rejections *and* SQLite execution errors) are returned as a
structured `{"error": ...}` payload rather than raised, so the agent can read the
message and self-correct instead of crashing. The guard is covered by an
extensive parametrized test suite in `tests/test_guard.py`.

---

## How to run

Prerequisites: Python 3.11 and [Ollama](https://ollama.com) installed.

```bash
cd q3-agentic-ai

# 1. Environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # adjust if your Ollama host/model differ

# 2. Pull the tool-calling model and start Ollama
ollama pull llama3.1:8b
ollama serve                    # (usually already running)

# 3. Seed the analytics database (idempotent)
python seed.py

# 4. Launch the Streamlit prototype
streamlit run app.py
```

Then ask questions such as:

- *Which product generated the most revenue from completed orders?*
- *How many completed orders did each country place?*
- *What is the average order value?*

The UI shows the step-by-step trace (thought → tool call → observation), the
generated SQL, the result table, an optional chart, and the final answer.

---

## Testing strategy

```bash
pytest -q                  # unit tests, no live LLM required (51 tests)
pytest -m integration      # optional live end-to-end test (needs Ollama)
```

- **No live LLM needed.** The agent loop is tested by injecting a scripted fake
  `LLMClient` (`ScriptedLLM`) that replays pre-baked tool calls and answers, so
  control flow is fully deterministic.
- **SQL safety guard** (`tests/test_guard.py`) — parametrized over many safe and
  unsafe queries (DELETE/UPDATE/DROP/ALTER/multi-statement/comment-smuggling).
- **Schema & query tools** (`tests/test_tools.py`) — run against a temp seeded
  SQLite DB: `list_tables`, `get_schema` (valid/invalid/injection), `run_sql`
  success/DML-rejection/bad-column, row-limit truncation, `make_chart`.
- **Agent loop control flow** (`tests/test_loop.py`) — asserts the loop calls the
  right tools in order, terminates on a final answer, **self-corrects** after a
  bad query, respects the guard mid-loop, and enforces the iteration cap.
- **Optional integration** (`tests/test_integration.py`) — marked
  `@pytest.mark.integration` and auto-skipped when Ollama is unreachable.

Every test uses a temporary, freshly seeded database (`conftest.py`) — the real
`analytics.db` is never touched.

---

## Limitations

- **Small local model.** `llama3.1:8b` is solid at tool calling but can still
  occasionally write suboptimal SQL or stop early; the iteration cap and
  self-correction loop mitigate but do not eliminate this.
- **Read-only by design.** The agent cannot modify data — intentional, but it
  means write-style questions are answered with explanation, not action.
- **Single database, no auth.** There is no row-level security or multi-tenant
  isolation; the guard protects against injection/mutation, not against a user
  who is legitimately allowed to query everything.
- **Chart tool is intentionally simple.** It returns a spec (labels/values) that
  the UI renders with native Streamlit charts; it is not a full visualisation
  grammar.
- **No long-term memory.** Memory is per-question (the message list); the agent
  does not persist learnings across sessions.

---

## Project layout

```
q3-agentic-ai/
├── agent/
│   ├── __init__.py
│   ├── config.py        # .env-driven typed settings
│   ├── db.py            # schema + idempotent seed + connection helpers
│   ├── tools.py         # tool fns + JSON schemas + SQL safety guard
│   ├── llm.py           # Ollama tool-calling client (+ fake-able protocol)
│   └── loop.py          # the agentic ReAct tool-calling loop
├── tests/
│   ├── conftest.py      # temp seeded DB + toolbox fixtures
│   ├── test_guard.py    # SQL safety guard
│   ├── test_tools.py    # schema/query/chart tools
│   ├── test_loop.py     # loop control flow + self-correction (mocked LLM)
│   └── test_integration.py  # optional live Ollama test
├── app.py               # Streamlit prototype
├── seed.py              # idempotent DB seeding entry point
├── requirements.txt
├── pytest.ini
├── .env.example
├── README.md
└── REQUIREMENTS.md
```

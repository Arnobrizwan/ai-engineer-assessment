# AI Engineer Assessment — Agentic RAG · Streaming Chat · Agentic AI

A single monorepo that answers **all three** assessment questions at an advanced level, end‑to‑end, running **100% locally** on an open‑source LLM (no paid API keys required).

| # | Project | What it is | Stack | Folder |
|---|---------|------------|-------|--------|
| **Q1** | **Agentic RAG** | A self‑grading retrieval agent that rewrites queries, retrieves with hybrid search, checks its own evidence, and answers **with citations** | Streamlit · Chroma · BM25 · RRF · Ollama | [`q1-agentic-rag/`](./q1-agentic-rag) |
| **Q2** | **Streaming Chat** | A FastAPI **SSE** endpoint that streams the LLM **token‑by‑token**, with DB‑backed chat memory and a polished web UI | FastAPI · SSE · SQLAlchemy/SQLite · Docker | [`q2-streaming-chat/`](./q2-streaming-chat) |
| **Q3** | **Agentic AI (SQL Analyst)** | A ReAct tool‑calling agent that answers business questions against a SQL database, self‑correcting on bad queries | Streamlit · Ollama tool‑calling · SQLite | [`q3-agentic-ai/`](./q3-agentic-ai) |

> **One engine for everything:** [Ollama](https://ollama.com) running `llama3.1:8b` (generation) and `nomic-embed-text` (embeddings). Everything is configured via `.env`; **no secrets are committed**.

---

## TL;DR — run it

```bash
# 0. Prerequisites: Python 3.11, Ollama installed and running
ollama pull llama3.1:8b
ollama pull nomic-embed-text          # only needed for Q1

# 1. Pick a project, create a venv, install, run
cd q1-agentic-rag   && python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && streamlit run app.py
cd q2-streaming-chat && python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && uvicorn app.main:app --reload   # then open http://localhost:8000
cd q3-agentic-ai    && python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && python seed.py && streamlit run app.py
```

Each subproject has its own deep‑dive `README.md` and a `REQUIREMENTS.md` that maps **every** assessment requirement (and bonus) to the exact file and function that satisfies it.

---

## Test status (all offline, LLM mocked)

| Project | Unit tests | Live integration test |
|---------|-----------|------------------------|
| Q1 Agentic RAG | **30 passed** | live hybrid retrieval; eval harness: **100% hit‑rate, 100% groundedness** |
| Q2 Streaming Chat | **19 passed** | real SSE token stream against live Ollama |
| Q3 Agentic AI | **51 passed** | live tool‑calling loop against `llama3.1:8b` (3/3 stable) |

> **100 unit tests, all green, no LLM required.** Each project also has one live `@pytest.mark.integration` test that runs end‑to‑end against Ollama and was verified passing during development.

```bash
cd <project> && . .venv/bin/activate && pytest -q          # unit suite, no LLM needed
pytest -q -m integration                                   # live test, auto-skips if Ollama is down
```

The whole suite is built so **CI can run it without a GPU or an API key** — every LLM/embedding call is injectable and mocked in unit tests. The single `@pytest.mark.integration` test per project is the only one that touches a live model, and it self‑skips when Ollama is unreachable. See each project's README "Testing strategy" section for the philosophy.

---

## Cross‑cutting investigation (presentation material)

### Traditional RAG vs Agentic RAG  *(full version in [`q1-agentic-rag/README.md`](./q1-agentic-rag/README.md))*

| Dimension | Traditional RAG | Agentic RAG (this repo) |
|-----------|-----------------|--------------------------|
| Control flow | Fixed: `embed → retrieve → stuff → generate` | Dynamic loop the LLM steers |
| Query | Used verbatim | Analyzed & **rewritten/decomposed** |
| Retrieval | One shot, top‑k dense | **Hybrid** (dense + BM25 + RRF), **re‑retrieves** if weak |
| Quality control | None — trusts whatever was retrieved | **Self‑grades** each chunk for relevance before answering |
| Failure mode | Hallucinates over irrelevant context | Says "insufficient evidence" / reformulates |
| Citations | Often bolted on | First‑class: every claim maps to `[doc p.X / chunk N]` |
| Cost / latency | Low, predictable | Higher, adaptive (only spends extra steps when needed) |

### Agentic AI — core components & characteristics  *(full version in [`q3-agentic-ai/README.md`](./q3-agentic-ai/README.md))*

- **Core components:** *Brain* (the LLM planner) · *Tools* (schema introspection, sandboxed `run_sql`, charting) · *Memory* (conversation + observation history) · *Planning* (ReAct: thought → action → observation) · *Orchestration loop* (bounded iterations, termination).
- **Key characteristics demonstrated:** *autonomy* (decides which tools to call), *reactivity* (responds to query + tool outputs), *tool use* (function calling), *goal‑directedness* (keeps acting until the question is answered), and **self‑correction** (feeds SQL errors back and retries).
- **Safety:** `run_sql` is read‑only by construction — comment stripping, single‑statement enforcement, SELECT/WITH‑only, keyword denylist, and a row cap.

---

## Repository layout

```
ai-engineer-assessment/
├── README.md                 # ← you are here (overview + presentation map)
├── PRESENTATION.md           # 15–20 min demo script / talk track
├── LICENSE
├── q1-agentic-rag/           # Question 1 — self-contained
├── q2-streaming-chat/        # Question 2 — self-contained
└── q3-agentic-ai/            # Question 3 — self-contained
```

## Design principles shared across all three

1. **Local‑first & reproducible** — one open‑source model engine, `.env` config, no vendor lock‑in, no committed secrets.
2. **Testable without an LLM** — deterministic unit tests via dependency injection + mocked models, so quality is provable in CI.
3. **Transparent agents** — every agent surfaces its reasoning trace (steps, tool calls, retrieved evidence) in the UI, not just a final answer.
4. **Requirement traceability** — each project ships a `REQUIREMENTS.md` mapping the brief to code.

See [`PRESENTATION.md`](./PRESENTATION.md) for the suggested demo flow.

# Presentation & Demo Guide (15–20 minutes)

A suggested talk track for demoing all three projects. Times are a guide; the live demos are the centerpiece.

## 0 · Setup before you present (do this once, off-camera)
```bash
ollama serve            # ensure the daemon is running
ollama pull llama3.1:8b
ollama pull nomic-embed-text
# create venvs + install for each project (see each README), and for Q3: python seed.py
```
Have three terminals (or tabs) ready, one per project.

---

## 1 · Framing (2 min)
- **Thesis:** all three problems share one backbone — a local open‑source LLM (`llama3.1:8b` via Ollama) — and differ in the *control structure* wrapped around it: a retrieval agent, a streaming service, and a tool‑using agent.
- **Why local:** reproducible, no API keys, no data leaving the machine, free to grade. Everything is `.env`‑configured; swapping to a hosted model is a one‑line change.
- Show the top‑level `README.md` table and the per‑project `REQUIREMENTS.md` traceability matrices.

## 2 · Q1 — Agentic RAG (5 min)
**Run:** `cd q1-agentic-rag && streamlit run app.py`
1. Ask a question answerable from the bundled docs (e.g. *"What pigment drives photosynthesis?"*). Show the **final answer with inline citations** and the exact **source passages** panel.
2. Open the **agent trace**: query analysis → hybrid retrieval (dense + BM25 + RRF) → **per‑chunk relevance grading** → answer.
3. Ask something the docs *don't* cover → the agent **reformulates / re‑retrieves**, then says **"insufficient evidence"** instead of hallucinating. This is the agentic difference.
4. **Talking point — Traditional vs Agentic RAG** (table in README): fixed pipeline vs. a loop that rewrites queries, self‑grades evidence, and re‑retrieves.
5. Show the **eval harness**: `python -m eval.eval` → retrieval hit‑rate + groundedness numbers. "This is how I prove retrieval quality, not just vibes."

## 3 · Q2 — Streaming Chat (4 min)
**Run:** `cd q2-streaming-chat && uvicorn app.main:app --reload` → open http://localhost:8000
1. Type a message → tokens **stream in live** (typing effect). Open DevTools → Network → the `/api/chat` request is `text/event-stream` (SSE), `data:` frames arriving incrementally.
2. **Memory demo:** "My name is Arnob." → then "What's my name?" → it remembers, because prior turns are loaded from SQLite and replayed into the prompt. Show the `messages` table.
3. Click **New session** → memory resets.
4. **Talking point:** the request lifecycle (browser → `POST /api/chat` → load history → stream from Ollama token‑by‑token → persist reply). Mention the **Dockerfile / docker‑compose** for one‑command deploy.

## 4 · Q3 — Agentic AI / SQL Analyst (5 min)
**Run:** `cd q3-agentic-ai && streamlit run app.py`
1. Ask a business question: *"What were our top 3 products by revenue?"* Watch the **ReAct trace**: thought → `list_tables` → `get_schema` → `run_sql(...)` → observation → final answer + result table + chart.
2. **Self‑correction demo:** ask something that makes the model write slightly wrong SQL first — show it getting an error observation and **retrying with corrected SQL**.
3. **Safety:** explain the `run_sql` guard — read‑only, single‑statement, SELECT/WITH only, keyword denylist, row cap. Try to get it to delete data → blocked.
4. **Talking point — Agentic AI investigation** (README section): core components (brain, tools, memory, planning, loop) and characteristics (autonomy, reactivity, tool use, goal‑directedness, self‑correction).

## 5 · Quality & testing (2 min)
- `pytest -q` in each project → **88 unit tests**, all green, **no LLM required** (models are dependency‑injected and mocked). This is what makes the suite CI‑friendly.
- One `@pytest.mark.integration` test per project hits the **live** model and **auto‑skips** if Ollama is down.
- Point at the live integration runs: Q1 dense retrieval, Q2 real SSE stream, Q3 real tool‑calling loop — all exercised end‑to‑end against `llama3.1:8b`.

## 6 · Wrap (1 min)
- Recap the three control structures over one engine.
- Trade‑offs & next steps: context windowing/summarization for long chats, reranker model for Q1, multi‑agent planning for Q3, hosted‑model swap for scale.

---

### Quick command cheat‑sheet
```bash
# Q1
cd q1-agentic-rag && . .venv/bin/activate && streamlit run app.py && python -m eval.eval
# Q2
cd q2-streaming-chat && . .venv/bin/activate && uvicorn app.main:app --reload   # http://localhost:8000
# Q3
cd q3-agentic-ai && . .venv/bin/activate && python seed.py && streamlit run app.py
# Tests (any project)
pytest -q ; pytest -q -m integration
```

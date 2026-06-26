# Q2 — Advanced Streaming LLM Chat

A production-quality, single-endpoint streaming chat application built with
**FastAPI** + **Ollama** (`llama3.1:8b`). The LLM response is streamed
**token-by-token** to the browser over **Server-Sent Events (SSE)**, rendered
live with a typing effect. Conversations are persisted to **SQLite** via
**SQLAlchemy**, so the model remembers prior turns within a session.

---

## Features

- **Token-by-token streaming** over SSE (`text/event-stream`) via FastAPI
  `StreamingResponse`, forwarding tokens from Ollama as they arrive.
- **Session memory**: every turn is persisted; full prior context is replayed
  to the model on each request.
- **Clean single-page UI** (`static/index.html`, vanilla JS): message bubbles,
  auto-scroll, live typing cursor, loading state, Markdown rendering, and a
  **New session** button.
- **REST API**: stream chat, fetch history, reset/delete a session, health probe.
- **Fully tested without a live LLM** (Ollama is mocked); one optional
  integration test auto-skips when Ollama is down.
- **Docker** + **docker-compose** for one-command deployment.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          Browser (static/index.html)                   │
│   message bubbles · live typing · markdown · auto-scroll · new session │
└───────────────┬───────────────────────────────▲───────────────────────┘
                │ POST /api/chat (JSON)          │ SSE: data: {token} …
                ▼                                 │
┌──────────────────────────────────────────────────────────────────────┐
│                       FastAPI app  (app/main.py)                       │
│  1. persist user message      ┌───────────────┐                        │
│  2. load ordered history ───► │  SQLite (DB)   │  app/db.py (SQLAlchemy)│
│  3. build_context()           │ sessions /     │                        │
│  4. stream from Ollama        │ messages       │                        │
│  5. persist assistant reply ► └───────────────┘                        │
└───────────────┬───────────────────────────────▲───────────────────────┘
                │ POST /api/chat (stream=true)    │ NDJSON token stream
                ▼                                 │
┌──────────────────────────────────────────────────────────────────────┐
│                 Ollama  (app/llm.py → /api/chat)  llama3.1:8b           │
└──────────────────────────────────────────────────────────────────────┘
```

### How SSE token streaming works (sequence)

1. **Browser → `POST /api/chat`** with `{message, session_id}`.
2. FastAPI **persists the user message** and **loads the full ordered history**
   from SQLite.
3. `build_context()` assembles `[system prompt, ...prior turns, new user msg]`.
4. FastAPI opens a streaming `POST` to **Ollama** (`/api/chat`, `stream=true`),
   which returns **newline-delimited JSON** — one token fragment per line.
5. Each fragment is wrapped as an SSE frame `data: {"type":"token","content":"…"}\n\n`
   and **flushed to the browser immediately** via `StreamingResponse`.
6. The browser reads the response body incrementally (`ReadableStream` reader),
   parses each frame, and appends the token to the assistant bubble (typing
   effect + auto-scroll).
7. On completion the server **persists the full assistant reply** and sends a
   final `data: {"type":"done", ...}` frame so the turn becomes memory for the
   next request.

The SSE frame protocol uses three message types: `start`, `token`, `done`
(plus `error` if the LLM call fails mid-stream).

---

## Database schema

`app/db.py` defines two tables (SQLAlchemy 2.0 typed ORM):

**`sessions`**

| column      | type      | notes                         |
|-------------|-----------|-------------------------------|
| `id`        | `String`  | primary key (client-supplied) |
| `created_at`| `DateTime`| UTC, defaulted                |

**`messages`**

| column       | type      | notes                                   |
|--------------|-----------|-----------------------------------------|
| `id`         | `Integer` | primary key, autoincrement              |
| `session_id` | `String`  | FK → `sessions.id`, indexed, cascade    |
| `role`       | `String`  | `user` / `assistant` / `system`         |
| `content`    | `Text`    | message body                            |
| `created_at` | `DateTime`| UTC, used for chronological ordering    |

History is loaded ordered by `(created_at, id)` for deterministic replay.

---

## API reference

| Method & path              | Body / query                       | Description                                 |
|----------------------------|------------------------------------|---------------------------------------------|
| `POST /api/chat`           | `{message, session_id?}`           | Stream assistant reply as SSE tokens        |
| `GET /api/history`         | `?session_id=`                     | Ordered message history for a session       |
| `POST /api/session/reset`  | `{session_id}`                     | Delete all messages for a session           |
| `DELETE /api/session`      | `?session_id=`                     | RESTful alias for reset                     |
| `GET /healthz`             | —                                  | Liveness probe (`{status, model}`)          |
| `GET /`                    | —                                  | Serves the chat UI                          |

`session_id` defaults to `"default"`, so the API works with zero client state
while still supporting multiple sessions.

---

## Run locally

Prerequisites: Python 3.11 and [Ollama](https://ollama.com) running with the
model pulled:

```bash
ollama pull llama3.1:8b
ollama serve        # usually already running on :11434
```

Then:

```bash
cd q2-streaming-chat
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # adjust if needed
uvicorn app.main:app --reload --port 8000
```

Open <http://localhost:8000> in your browser.

---

## Run with Docker

The container runs the FastAPI app and talks to **Ollama on the host** via
`host.docker.internal` (configurable through `OLLAMA_BASE_URL`).

```bash
cd q2-streaming-chat
docker compose up --build
```

Then open <http://localhost:8000>. The SQLite database is persisted in the
named volume `chat-data` (`/data/chat.db` inside the container).

> On Linux, `host.docker.internal` is wired up through the `extra_hosts:
> host-gateway` entry already present in `docker-compose.yml`. Ensure Ollama is
> listening on all interfaces if needed (`OLLAMA_HOST=0.0.0.0 ollama serve`).

---

## Testing & methodology

```bash
source .venv/bin/activate
pytest -q                 # unit + endpoint tests (no LLM needed)
pytest -m integration     # optional: real Ollama, auto-skips if down
```

The suite (19 tests) runs **without a live model** by monkeypatching the Ollama
streaming client. Coverage:

- **`tests/test_db.py`** — session creation, message append (auto-creates
  session), history ordering, session scoping, reset.
- **`tests/test_context.py`** — context assembly: system prompt prepended and
  all prior turns included in order.
- **`tests/test_llm.py`** — NDJSON stream parsing into tokens, tolerance of
  malformed/keep-alive lines, stopping at the `done` marker.
- **`tests/test_api.py`** — the SSE endpoint via FastAPI `TestClient`
  (asserts `text/event-stream` and `start`/`token`/`done` `data:` frames with a
  mocked Ollama), user + assistant persistence, prior-turn context propagation,
  history retrieval, reset/delete, input validation, health probe, and one
  `@pytest.mark.integration` end-to-end test that auto-skips when Ollama is down.

Each test uses an **isolated temp SQLite database** (via the `tmp_path`
fixture); the app's `get_db` dependency is overridden so the production
`chat.db` is never touched.

---

## Project layout

```
q2-streaming-chat/
├── app/
│   ├── __init__.py
│   ├── config.py        # .env-driven settings (python-dotenv)
│   ├── db.py            # SQLAlchemy models + persistence helpers
│   ├── llm.py           # Ollama streaming client (NDJSON → tokens)
│   ├── schemas.py       # Pydantic request/response models
│   └── main.py          # FastAPI app, SSE route, history/reset, static UI
├── static/
│   └── index.html       # single-page chat UI (vanilla JS, SSE consumer)
├── tests/               # pytest suite (LLM mocked) + conftest fixtures
├── requirements.txt
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── pytest.ini
├── REQUIREMENTS.md      # assignment requirement → file/function mapping
└── README.md
```

---

## Limitations & notes

- **SQLite**: fine for a single-node demo; for concurrency/scale, swap
  `DATABASE_URL` for Postgres (the SQLAlchemy layer is portable).
- **Unbounded context**: every prior turn is replayed. For long sessions a
  sliding window or summarization step should cap the prompt size.
- **No auth**: there is no authentication or per-user isolation beyond the
  opaque `session_id`. Add auth before exposing publicly.
- **SSE is server→client only**: cancellation isn't surfaced back to Ollama mid
  generation; the stream simply ends when the client disconnects.
- The Markdown renderer in the UI is intentionally minimal (escape-first, then
  format) to avoid pulling in a heavy dependency; it is not a full CommonMark
  implementation.

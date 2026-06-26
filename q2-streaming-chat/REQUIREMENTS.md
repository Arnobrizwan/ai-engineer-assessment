# Requirement → Implementation Mapping

This document maps every assignment requirement (and bonus) to the exact file
and function/symbol that satisfies it.

## Core requirements

| # | Requirement | File · Symbol |
|---|-------------|---------------|
| 1 | One REST endpoint streaming the LLM response **token-by-token** | `app/main.py` · `chat()` → `event_generator()` returning `StreamingResponse(media_type="text/event-stream")` |
| 1 | SSE frame formatting (`data: …\n\n`) | `app/main.py` · `_sse()` |
| 1 | Stream tokens **as they arrive** from Ollama | `app/llm.py` · `stream_chat()` + `_iter_ollama_lines()` (httpx `client.stream` over `/api/chat`, `stream=true`) |
| 2 | Clean single-page chat UI consuming SSE with typing effect | `static/index.html` · `sendMessage()` (ReadableStream reader → `renderMarkdown`) |
| 2 | Message bubbles, auto-scroll, loading state | `static/index.html` · `addBubble()`, `scrollDown()`, `setStreaming()` |
| 2 | Markdown rendering | `static/index.html` · `renderMarkdown()` / `escapeHtml()` |
| 2 | "New session" button | `static/index.html` · `resetSession()` + `#reset` button |
| 3 | Chat session management with a database (SQLAlchemy + SQLite) | `app/db.py` · `ChatSession`, `Message`, `engine`, `SessionLocal` |
| 3 | Tables `sessions` and `messages` (role, content, created_at, session_id) | `app/db.py` · `ChatSession.__tablename__="sessions"`, `Message.__tablename__="messages"` |
| 3 | Load prior messages and send full context to the LLM (memory) | `app/db.py` · `get_history()`; `app/main.py` · `build_context()` |
| 3 | Persist new user message and streamed assistant reply | `app/main.py` · `chat()` (persist user before stream; persist reply in `finally`) via `app/db.py` · `add_message()` |
| 3 | Support a `session_id` | `app/schemas.py` · `ChatRequest.session_id` (default `"default"`) |
| 3 | `POST /api/chat` (stream) | `app/main.py` · `chat()` |
| 3 | `GET /api/history?session_id=` | `app/main.py` · `history()` |
| 3 | `POST /api/session/reset` / `DELETE` | `app/main.py` · `reset_session()`, `delete_session()` |
| 4 | DB persistence tests (create session, append, history ordering) | `tests/test_db.py` |
| 4 | Context assembly test (prior turns included) | `tests/test_context.py`, `tests/test_api.py::test_context_includes_prior_turns` |
| 4 | SSE endpoint test via `TestClient` asserting `data:` chunks, mocked Ollama | `tests/test_api.py::test_chat_streams_tokens` |
| 4 | History retrieval test | `tests/test_api.py::test_history_endpoint_returns_ordered_messages` |
| 4 | In-memory / temp SQLite DB in tests | `tests/conftest.py` · `engine` (tmp_path), `client` (dependency override) |
| 4 | Tests run WITHOUT a live LLM (mock/monkeypatch) | `tests/test_api.py` · `_fake_stream`; `tests/test_llm.py` · monkeypatched `_iter_ollama_lines` |
| 4 | Optional `@pytest.mark.integration`, auto-skipped if Ollama down | `tests/test_api.py::test_integration_real_ollama` + `app/llm.py` · `is_ollama_up()` |

## Bonus requirements

| Bonus | Requirement | File · Symbol |
|-------|-------------|---------------|
| Docker | `Dockerfile` running the FastAPI app | `Dockerfile` (uvicorn entrypoint) |
| Docker | `docker-compose.yml`, Ollama on host via `host.docker.internal` | `docker-compose.yml` (`OLLAMA_BASE_URL`, `extra_hosts: host-gateway`) |
| Docker | Configurable base URL | `app/config.py` · `Settings.ollama_base_url` (env `OLLAMA_BASE_URL`) |
| Friendly UI | User-friendly interface | `static/index.html` (bubbles, typing cursor, markdown, auto-scroll, new-session) |

## Configuration & infra

| Item | File · Symbol |
|------|---------------|
| `.env` via `python-dotenv`, `.env.example` provided | `app/config.py` · `load_dotenv()` / `get_settings()`; `.env.example` |
| Pinned dependencies | `requirements.txt` |
| Health probe for Docker `HEALTHCHECK` | `app/main.py` · `healthz()` |
| DB schema bootstrap on startup | `app/main.py` · `lifespan()` → `app/db.py` · `init_db()` |

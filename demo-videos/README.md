# Demo Videos

Screen recordings captured from the **real running prototypes** (live `llama3.1:8b` via Ollama). Silent demo footage — see [`../PRESENTATION.md`](../PRESENTATION.md) for the talk track.

> GitHub doesn't play `.mp4` inline in a README — click a file below, then press the **View raw** / download button to watch.

| Video | Length | What it shows |
|-------|--------|----------------|
| [`Q1-Agentic-RAG-demo.mp4`](./Q1-Agentic-RAG-demo.mp4) | ~62s | A grounded answer **with an inline citation + source passage + agent reasoning trace** (analyze → retrieve → grade), then an **out-of-scope** question where the agent refuses with *"insufficient evidence"* — the agentic difference. |
| [`Q2-Streaming-Chat-demo.mp4`](./Q2-Streaming-Chat-demo.mp4) | ~24s | **Token-by-token SSE streaming** plus **session memory** — the assistant is told a name and recalls it on a fresh request in the same session (replayed from SQLite). |
| [`Q3-SQL-Agent-demo.mp4`](./Q3-SQL-Agent-demo.mp4) | ~52s | The SQL agent answering a business question, showing the **thought → tool call → generated SQL → observation** trace, including **self-correction** after a bad query. |

The Q1 and Q3 clips are played back at 2× to trim local-model latency; Q2 is real-time. Recorded headlessly with Playwright/Chromium and encoded with ffmpeg.

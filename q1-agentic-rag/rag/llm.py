"""Thin Ollama client wrapper.

Exposes two capabilities used across the system:

* :meth:`LLMClient.chat` -- chat completion against ``llama3.1:8b``.
* :meth:`LLMClient.embed` -- text embeddings against ``nomic-embed-text``.

The client talks to Ollama's *native* API (``/api/chat`` and
``/api/embeddings``). It is intentionally small and dependency-light so that
unit tests can monkeypatch :meth:`chat` / :meth:`embed` without a live server.
"""

from __future__ import annotations

from typing import Sequence

import requests

from .config import Config, get_config


class LLMError(RuntimeError):
    """Raised when the Ollama server returns an error or is unreachable."""


class LLMClient:
    """Client for chat + embedding calls against a local Ollama server."""

    def __init__(self, config: Config | None = None, timeout: float = 120.0) -> None:
        self.config = config or get_config()
        self.timeout = timeout

    # ------------------------------------------------------------------ chat
    def chat(self, messages: Sequence[dict[str, str]], temperature: float = 0.0) -> str:
        """Return the assistant message content for a chat ``messages`` list.

        ``messages`` is a list of ``{"role": ..., "content": ...}`` dicts in the
        OpenAI/Ollama format. Uses greedy decoding by default for determinism.
        """
        url = f"{self.config.ollama_native_url}/api/chat"
        payload = {
            "model": self.config.llm_model,
            "messages": list(messages),
            "stream": False,
            "options": {"temperature": temperature},
        }
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:  # pragma: no cover - network
            raise LLMError(f"Ollama chat request failed: {exc}") from exc
        return data.get("message", {}).get("content", "").strip()

    # ------------------------------------------------------------- embeddings
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding vector per input string in ``texts``."""
        url = f"{self.config.ollama_native_url}/api/embeddings"
        vectors: list[list[float]] = []
        for text in texts:
            payload = {"model": self.config.embed_model, "prompt": text}
            try:
                resp = requests.post(url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as exc:  # pragma: no cover - network
                raise LLMError(f"Ollama embeddings request failed: {exc}") from exc
            vectors.append(data.get("embedding", []))
        return vectors

    def is_available(self) -> bool:
        """Return ``True`` if the Ollama server answers a health ping."""
        try:
            resp = requests.get(
                f"{self.config.ollama_native_url}/api/tags", timeout=2.0
            )
            return resp.status_code == 200
        except requests.RequestException:
            return False

"""Offline-friendly evaluation harness for the agentic RAG system.

Measures two things over ``eval/qa.jsonl`` against the bundled sample docs:

* **retrieval hit-rate** -- did retrieval surface a chunk from the expected
  source document?
* **answer groundedness** -- does the final answer contain the expected
  substring(s) AND cite at least one real source passage?

Runs offline by default using a deterministic mock LLM (no Ollama needed). Pass
``--live`` to evaluate against a running Ollama server instead.

Usage:
    python -m eval.eval            # offline / mock
    python -m eval.eval --live     # against live Ollama
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from rag.config import get_config  # noqa: E402
from rag.ingest import ingest_directory  # noqa: E402
from rag.pipeline import build_agent  # noqa: E402

QA_PATH = Path(__file__).resolve().parent / "qa.jsonl"


class MockLLM:
    """Deterministic stand-in so the eval runs without a live model.

    * rewrite/reformulate -> identity on the query.
    * grader -> 'yes' (defer relevance to retrieval quality).
    * answer -> echoes the provided sources verbatim (keeps citations + facts),
      which lets us measure whether retrieval actually surfaced the answer text.
    """

    def chat(self, messages: Sequence[dict[str, str]], temperature: float = 0.0) -> str:
        system = messages[0]["content"].lower()
        user = messages[-1]["content"]
        if "relevance grader" in system:
            return "yes"
        if "precise assistant" in system:
            return user  # the sources block contains markers + facts
        return user.strip()


@dataclass
class EvalItem:
    question: str
    expected_substrings: list[str]
    expected_source: str


def load_qa(path: Path = QA_PATH) -> list[EvalItem]:
    items: list[EvalItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        items.append(
            EvalItem(
                question=row["question"],
                expected_substrings=[s.lower() for s in row["expected_substrings"]],
                expected_source=row["expected_source"],
            )
        )
    return items


def _make_embed_fn(live: bool):
    if not live:
        return None
    from rag.llm import LLMClient

    client = LLMClient(get_config())
    if not client.is_available():
        raise SystemExit("Ollama is not reachable; cannot run --live eval.")
    return client.embed


def run_eval(live: bool = False) -> dict[str, float]:
    config = get_config()
    chunks = ingest_directory(config=config)

    if live:
        from rag.llm import LLMClient

        llm = LLMClient(config)
        embed_fn = llm.embed
    else:
        llm = MockLLM()
        embed_fn = None  # sparse-only retrieval, fully offline

    agent = build_agent(chunks, llm=llm, embed_fn=embed_fn, config=config)
    items = load_qa()

    retrieval_hits = 0
    grounded_hits = 0
    rows: list[dict] = []

    for item in items:
        result = agent.run(item.question)
        retrieved_docs = {p.chunk.doc_name for p in result.passages}
        hit = item.expected_source in retrieved_docs

        answer_l = result.answer.lower()
        substr_ok = any(s in answer_l for s in item.expected_substrings)
        cited_ok = bool(result.citations)
        grounded = substr_ok and cited_ok

        retrieval_hits += int(hit)
        grounded_hits += int(grounded)
        rows.append(
            {
                "question": item.question,
                "retrieval_hit": hit,
                "grounded": grounded,
                "citations": result.citations,
            }
        )

    n = len(items)
    metrics = {
        "n": n,
        "retrieval_hit_rate": retrieval_hits / n if n else 0.0,
        "groundedness": grounded_hits / n if n else 0.0,
    }

    print(f"\nEvaluation ({'live' if live else 'mock'}) over {n} questions")
    print("-" * 60)
    for row in rows:
        flag_r = "OK " if row["retrieval_hit"] else "MISS"
        flag_g = "OK " if row["grounded"] else "MISS"
        print(f"[retrieval {flag_r}] [grounded {flag_g}] {row['question']}")
    print("-" * 60)
    print(f"Retrieval hit-rate: {metrics['retrieval_hit_rate']:.2%}")
    print(f"Groundedness:       {metrics['groundedness']:.2%}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic RAG eval harness")
    parser.add_argument(
        "--live", action="store_true", help="Evaluate against a live Ollama server."
    )
    args = parser.parse_args()
    run_eval(live=args.live)


if __name__ == "__main__":
    main()

"""Offline-friendly evaluation harness for the agentic RAG system.

Measures, over ``eval/qa.jsonl`` against the bundled sample docs:

* **retrieval hit-rate** -- did retrieval surface a chunk from the expected
  source document?
* **MRR** -- mean reciprocal rank of the first relevant retrieved passage.
* **nDCG@k** -- normalised discounted cumulative gain over the ranked passages.
* **answer groundedness** -- does the final answer contain the expected
  substring(s) AND cite at least one real source passage?
* **faithfulness (LLM-judge)** -- RAGAS-style 0..1 score of how well the answer
  is supported by the retrieved context (``--live`` only; N/A in mock mode).

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

from eval.metrics import (  # noqa: E402
    mean_reciprocal_rank,
    ndcg_at_k,
    parse_score,
)
from rag.config import get_config  # noqa: E402
from rag.ingest import ingest_directory  # noqa: E402
from rag.pipeline import build_agent  # noqa: E402

QA_PATH = Path(__file__).resolve().parent / "qa.jsonl"
NDCG_K = 5

_FAITHFULNESS_SYS = (
    "You are a strict faithfulness judge. Given an ANSWER and the CONTEXT it "
    "was supposed to be grounded in, rate how well every claim in the answer is "
    "supported by the context. Reply with ONLY a single number between 0.0 "
    "(unsupported / hallucinated) and 1.0 (fully supported)."
)


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


def judge_faithfulness(llm, answer: str, context: str) -> float:
    """RAGAS-style LLM-as-judge faithfulness score in ``[0.0, 1.0]``.

    Asks the model how well ``answer`` is supported by ``context`` and parses a
    single float, defaulting to ``0.0`` on any parse failure. ``llm`` only needs
    a ``.chat(messages)`` method, so it is trivially mockable in tests.
    """
    if not answer.strip() or not context.strip():
        return 0.0
    reply = llm.chat(
        [
            {"role": "system", "content": _FAITHFULNESS_SYS},
            {
                "role": "user",
                "content": f"CONTEXT:\n{context}\n\nANSWER:\n{answer}\n\nScore:",
            },
        ]
    )
    return parse_score(reply)


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
    rr_scores: list[float] = []
    ndcg_scores: list[float] = []
    faithfulness_scores: list[float] = []
    rows: list[dict] = []

    for item in items:
        result = agent.run(item.question)

        # Binary relevances over the ranked retrieved passages.
        relevances = [
            int(p.chunk.doc_name == item.expected_source) for p in result.passages
        ]
        hit = any(relevances)
        rr = mean_reciprocal_rank(relevances)
        ndcg = ndcg_at_k(relevances, NDCG_K)

        answer_l = result.answer.lower()
        substr_ok = any(s in answer_l for s in item.expected_substrings)
        cited_ok = bool(result.citations)
        grounded = substr_ok and cited_ok

        # RAGAS-style faithfulness is LLM-judged -> live only.
        if live:
            context = "\n\n".join(p.text for p in result.passages)
            faithfulness_scores.append(
                judge_faithfulness(llm, result.answer, context)
            )

        retrieval_hits += int(hit)
        grounded_hits += int(grounded)
        rr_scores.append(rr)
        ndcg_scores.append(ndcg)
        rows.append(
            {
                "question": item.question,
                "retrieval_hit": hit,
                "grounded": grounded,
                "rr": rr,
                "ndcg": ndcg,
                "citations": result.citations,
            }
        )

    n = len(items)

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    metrics = {
        "n": n,
        "retrieval_hit_rate": retrieval_hits / n if n else 0.0,
        "mrr": _mean(rr_scores),
        f"ndcg@{NDCG_K}": _mean(ndcg_scores),
        "groundedness": grounded_hits / n if n else 0.0,
        "faithfulness": _mean(faithfulness_scores) if live else None,
    }

    print(f"\nEvaluation ({'live' if live else 'mock'}) over {n} questions")
    print("-" * 60)
    for row in rows:
        flag_r = "OK " if row["retrieval_hit"] else "MISS"
        flag_g = "OK " if row["grounded"] else "MISS"
        print(
            f"[retrieval {flag_r}] [grounded {flag_g}] "
            f"[rr {row['rr']:.2f}] [ndcg {row['ndcg']:.2f}] {row['question']}"
        )
    print("-" * 60)
    print(f"Retrieval hit-rate:    {metrics['retrieval_hit_rate']:.2%}")
    print(f"MRR:                   {metrics['mrr']:.3f}")
    print(f"nDCG@{NDCG_K}:               {metrics[f'ndcg@{NDCG_K}']:.3f}")
    print(f"Groundedness:          {metrics['groundedness']:.2%}")
    if live:
        print(f"Faithfulness (LLM-judge): {metrics['faithfulness']:.3f}")
    else:
        print("Faithfulness (LLM-judge): N/A (mock mode; use --live)")
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

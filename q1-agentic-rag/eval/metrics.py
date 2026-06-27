"""Pure, unit-testable retrieval-ranking and faithfulness metrics.

These functions take an ordered list of binary relevances (1 = relevant,
0 = not), so they are LLM-free and deterministic. The LLM-as-judge faithfulness
metric is kept separate (it needs a model) and is only invoked in ``--live`` eval.
"""

from __future__ import annotations

import math
import re
from typing import Sequence


def mean_reciprocal_rank(ranked_relevances: Sequence[int]) -> float:
    """Reciprocal rank of the first relevant item (0.0 if none).

    For a single ranked list this is the Reciprocal Rank; averaging it across
    questions (done by the harness) yields the Mean Reciprocal Rank (MRR).

    Args:
        ranked_relevances: binary relevances in rank order (best first).

    Returns:
        ``1 / rank_of_first_relevant`` (1-based), or ``0.0`` if nothing is
        relevant.
    """
    for index, rel in enumerate(ranked_relevances, start=1):
        if rel:
            return 1.0 / index
    return 0.0


def dcg_at_k(ranked_relevances: Sequence[int], k: int) -> float:
    """Discounted Cumulative Gain over the top ``k`` items (binary gains)."""
    dcg = 0.0
    for index, rel in enumerate(ranked_relevances[:k], start=1):
        # Standard DCG: gain / log2(rank + 1).
        dcg += rel / math.log2(index + 1)
    return dcg


def ndcg_at_k(ranked_relevances: Sequence[int], k: int = 5) -> float:
    """Normalised DCG@k with binary gains (0.0 when no relevant items exist).

    The ideal ranking places all relevant items first; nDCG = DCG / IDCG.
    """
    actual = dcg_at_k(ranked_relevances, k)
    ideal_order = sorted(ranked_relevances, reverse=True)
    ideal = dcg_at_k(ideal_order, k)
    if ideal == 0.0:
        return 0.0
    return actual / ideal


_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+")


def parse_score(text: str) -> float:
    """Robustly parse a 0.0-1.0 float from a model reply (0.0 on failure).

    Accepts replies like ``"0.8"``, ``"Score: 0.75"`` or ``"1"``; clamps to
    ``[0, 1]`` and returns ``0.0`` if no number is found.
    """
    match = _FLOAT_RE.search(text or "")
    if not match:
        return 0.0
    try:
        value = float(match.group())
    except ValueError:
        return 0.0
    return max(0.0, min(1.0, value))

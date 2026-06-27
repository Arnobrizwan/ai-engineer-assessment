"""Evaluation harness for the SQL Analytics Agent.

Provides a small, dependency-light accuracy benchmark: a set of natural-language
business questions (``qa.jsonl``) whose ground-truth answers were derived by
querying the seeded ``analytics.db`` directly, plus a runner (``eval.py``) that
scores the live agent's answers and reports an overall accuracy percentage and
average iteration count.
"""

from __future__ import annotations

__all__ = [
    "EvalCase",
    "QuestionResult",
    "EvalReport",
    "answer_matches",
    "aggregate",
    "load_cases",
]

from .eval import (  # noqa: E402  (re-export convenience)
    EvalCase,
    EvalReport,
    QuestionResult,
    aggregate,
    answer_matches,
    load_cases,
)

"""Accuracy evaluation harness for the SQL Analytics Agent.

Runs the **real** agent (against a live Ollama server) over a fixed set of
business questions whose ground-truth answers were derived directly from the
seeded ``analytics.db``. For each question it checks whether the agent's final
answer contains at least one expected substring (case-insensitive), then reports
per-question PASS/FAIL, the overall **SQL-answer accuracy** (e.g. ``9/10 =
90%``), and the **average number of agent iterations**.

Run it as a module (requires a running Ollama with the configured model):

    python -m eval.eval

The scoring and aggregation logic is factored into small pure functions
(:func:`answer_matches`, :func:`aggregate`) so it can be unit-tested offline
without a live LLM (see ``tests/test_eval.py``).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

QA_PATH = Path(__file__).resolve().parent / "qa.jsonl"


@dataclass(frozen=True)
class EvalCase:
    """A single evaluation question with its accepted answer substrings.

    Attributes:
        id: Stable identifier for the question.
        question: The natural-language business question posed to the agent.
        expected_substrings: Answer passes if it contains at least one of these
            (matched case-insensitively).
    """

    id: str
    question: str
    expected_substrings: tuple[str, ...]


@dataclass
class QuestionResult:
    """The graded outcome of running one :class:`EvalCase`.

    Attributes:
        case: The evaluation case that was run.
        answer: The agent's final natural-language answer.
        passed: Whether the answer matched an expected substring.
        iterations: Number of agent loop iterations taken.
        sql: The last SQL the agent executed (if any), for debugging.
    """

    case: EvalCase
    answer: str
    passed: bool
    iterations: int
    sql: str | None = None


@dataclass
class EvalReport:
    """Aggregate results across all questions.

    Attributes:
        results: Per-question graded results.
        total: Number of questions evaluated.
        passed: Number of questions answered correctly.
        accuracy: Fraction correct in ``[0, 1]``.
        avg_iterations: Mean agent iterations across all questions.
    """

    results: list[QuestionResult] = field(default_factory=list)
    total: int = 0
    passed: int = 0
    accuracy: float = 0.0
    avg_iterations: float = 0.0


def answer_matches(answer: str, expected_substrings: Iterable[str]) -> bool:
    """Return ``True`` if ``answer`` contains any expected substring.

    Matching is case-insensitive and whitespace-insensitive at the edges. An
    empty ``expected_substrings`` never matches.

    Args:
        answer: The agent's final answer text.
        expected_substrings: Acceptable substrings; one match suffices.

    Returns:
        ``True`` if at least one substring is present, else ``False``.
    """
    if not answer:
        return False
    haystack = answer.lower()
    for expected in expected_substrings:
        needle = str(expected).strip().lower()
        if needle and needle in haystack:
            return True
    return False


def aggregate(results: Sequence[QuestionResult]) -> EvalReport:
    """Aggregate per-question results into an :class:`EvalReport`.

    Args:
        results: The graded per-question results.

    Returns:
        An :class:`EvalReport` with totals, accuracy, and average iterations.
        For an empty input, accuracy and average iterations are ``0.0``.
    """
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    accuracy = (passed / total) if total else 0.0
    avg_iterations = (
        sum(r.iterations for r in results) / total if total else 0.0
    )
    return EvalReport(
        results=list(results),
        total=total,
        passed=passed,
        accuracy=accuracy,
        avg_iterations=avg_iterations,
    )


def load_cases(path: Path | str = QA_PATH) -> list[EvalCase]:
    """Load evaluation cases from a JSONL file.

    Args:
        path: Path to the ``qa.jsonl`` file.

    Returns:
        A list of :class:`EvalCase` objects in file order.

    Raises:
        ValueError: If a line is missing required fields.
    """
    cases: list[EvalCase] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            try:
                cases.append(
                    EvalCase(
                        id=obj["id"],
                        question=obj["question"],
                        expected_substrings=tuple(obj["expected_substrings"]),
                    )
                )
            except KeyError as exc:  # pragma: no cover - defensive
                raise ValueError(
                    f"{path}:{line_no} missing required field {exc}"
                ) from exc
    return cases


def format_report(report: EvalReport) -> str:
    """Render a human-readable summary of an :class:`EvalReport`.

    Args:
        report: The aggregated report.

    Returns:
        A multi-line string suitable for printing to a terminal.
    """
    lines = ["", "SQL Analytics Agent — Evaluation", "=" * 40]
    for result in report.results:
        status = "PASS" if result.passed else "FAIL"
        lines.append(f"[{status}] {result.case.id}: {result.case.question}")
        lines.append(f"        answer: {result.answer[:160]}")
        if not result.passed:
            lines.append(
                f"        expected one of: {list(result.case.expected_substrings)}"
            )
    pct = report.accuracy * 100
    lines.append("-" * 40)
    lines.append(
        f"SQL-answer accuracy: {report.passed}/{report.total} = {pct:.0f}%"
    )
    lines.append(f"Average iterations:  {report.avg_iterations:.2f}")
    lines.append("")
    return "\n".join(lines)


def run_evaluation() -> EvalReport:
    """Run the live agent over every case and return the aggregated report.

    This imports the agent stack lazily so that importing this module (e.g. for
    unit-testing the pure scoring functions) never requires the agent
    dependencies or a live LLM.

    Returns:
        The aggregated :class:`EvalReport`.
    """
    # Lazy imports keep the pure-function path import-light and network-free.
    from agent.config import get_settings
    from agent.db import ensure_seeded
    from agent.llm import OllamaClient
    from agent.loop import Agent
    from agent.tools import Toolbox

    settings = get_settings()
    ensure_seeded(settings.db_path)
    toolbox = Toolbox(settings.db_path, row_limit=settings.sql_row_limit)
    agent = Agent(OllamaClient(settings), toolbox, settings)

    results: list[QuestionResult] = []
    for case in load_cases():
        outcome = agent.run(case.question)
        passed = answer_matches(outcome.answer, case.expected_substrings)
        results.append(
            QuestionResult(
                case=case,
                answer=outcome.answer,
                passed=passed,
                iterations=outcome.iterations,
                sql=outcome.last_sql,
            )
        )
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {case.id} ({outcome.iterations} iters)")

    return aggregate(results)


def main() -> int:
    """CLI entry point. Returns a process exit code.

    Returns:
        ``0`` on a completed evaluation, ``2`` if Ollama is unavailable.
    """
    from agent.config import get_settings
    from agent.llm import is_ollama_available

    settings = get_settings()
    if not is_ollama_available(settings):
        print(
            "Ollama is not reachable at "
            f"{settings.ollama_base_url}. Start it with `ollama serve` and pull "
            f"the model: `ollama pull {settings.llm_model}`. Skipping evaluation.",
            file=sys.stderr,
        )
        return 2

    report = run_evaluation()
    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

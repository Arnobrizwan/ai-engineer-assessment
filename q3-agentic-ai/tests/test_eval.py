"""Offline unit tests for the evaluation scoring/aggregation functions.

These never touch a live LLM: they exercise the pure functions
(:func:`answer_matches`, :func:`aggregate`, :func:`load_cases`) with synthetic
data and a mocked agent answer set.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.eval import (
    EvalCase,
    QuestionResult,
    aggregate,
    answer_matches,
    format_report,
    load_cases,
)


# -- answer_matches --------------------------------------------------------


def test_answer_matches_case_insensitive() -> None:
    assert answer_matches("The top product is the 27-INCH monitor.", ["27-inch Monitor"])


def test_answer_matches_any_of_several() -> None:
    assert answer_matches("It is the Monitor.", ["Display", "Monitor"])


def test_answer_matches_substring_in_number() -> None:
    assert answer_matches("There are 8 customers.", ["8"])


def test_answer_no_match() -> None:
    assert answer_matches("There are 9 customers.", ["8"]) is False


def test_answer_empty_inputs() -> None:
    assert answer_matches("", ["8"]) is False
    assert answer_matches("something", []) is False
    assert answer_matches("something", [""]) is False


# -- aggregate -------------------------------------------------------------


def _case(cid: str) -> EvalCase:
    return EvalCase(id=cid, question="q?", expected_substrings=("x",))


def _result(cid: str, passed: bool, iterations: int) -> QuestionResult:
    return QuestionResult(
        case=_case(cid), answer="a", passed=passed, iterations=iterations
    )


def test_aggregate_known_accuracy() -> None:
    # 9 pass / 1 fail => 90% accuracy.
    results = [_result(f"q{i}", True, 3) for i in range(9)]
    results.append(_result("q9", False, 5))
    report = aggregate(results)

    assert report.total == 10
    assert report.passed == 9
    assert report.accuracy == pytest.approx(0.9)
    # Average iterations = (9*3 + 5) / 10 = 3.2
    assert report.avg_iterations == pytest.approx(3.2)


def test_aggregate_all_pass() -> None:
    results = [_result(f"q{i}", True, 2) for i in range(4)]
    report = aggregate(results)
    assert report.accuracy == pytest.approx(1.0)
    assert report.avg_iterations == pytest.approx(2.0)


def test_aggregate_empty() -> None:
    report = aggregate([])
    assert report.total == 0
    assert report.accuracy == 0.0
    assert report.avg_iterations == 0.0


def test_format_report_contains_accuracy() -> None:
    results = [_result("q0", True, 1), _result("q1", False, 2)]
    text = format_report(aggregate(results))
    assert "1/2 = 50%" in text
    assert "Average iterations" in text


# -- load_cases (uses the real qa.jsonl shipped with the package) ----------


def test_load_cases_from_shipped_file() -> None:
    cases = load_cases()
    assert len(cases) >= 10
    ids = {c.id for c in cases}
    assert "q01_customer_count" in ids
    for case in cases:
        assert case.question
        assert case.expected_substrings  # non-empty


def test_load_cases_grading_with_mocked_answers() -> None:
    """Simulate an agent answering: grade a mix of right/wrong answers."""
    cases = load_cases()
    # Mock: answer the first case correctly, the rest with a wrong sentinel.
    mocked_answers = {cases[0].id: cases[0].expected_substrings[0]}
    results = [
        QuestionResult(
            case=c,
            answer=mocked_answers.get(c.id, "definitely-wrong-answer-zzz"),
            passed=answer_matches(
                mocked_answers.get(c.id, "definitely-wrong-answer-zzz"),
                c.expected_substrings,
            ),
            iterations=4,
        )
        for c in cases
    ]
    report = aggregate(results)
    assert report.passed == 1
    assert report.total == len(cases)

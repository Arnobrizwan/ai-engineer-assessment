"""Deterministic offline unit tests for eval ranking metrics + score parsing."""

from __future__ import annotations

import math

from eval.metrics import (
    dcg_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    parse_score,
)


# ------------------------------------------------------------------------ MRR
def test_mrr_first_relevant_at_rank_one():
    assert mean_reciprocal_rank([1, 0, 0]) == 1.0


def test_mrr_first_relevant_at_rank_two():
    assert mean_reciprocal_rank([0, 1, 0]) == 0.5


def test_mrr_first_relevant_at_rank_three():
    assert math.isclose(mean_reciprocal_rank([0, 0, 1]), 1.0 / 3.0)


def test_mrr_no_relevant_is_zero():
    assert mean_reciprocal_rank([0, 0, 0]) == 0.0


def test_mrr_uses_first_relevant_only():
    assert mean_reciprocal_rank([0, 1, 1]) == 0.5


# ---------------------------------------------------------------------- nDCG
def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k([1, 1, 0, 0], k=5) == 1.0


def test_ndcg_no_relevant_is_zero():
    assert ndcg_at_k([0, 0, 0], k=5) == 0.0


def test_ndcg_single_relevant_at_rank_two():
    # DCG = 1/log2(3); IDCG = 1/log2(2) = 1 -> nDCG = 1/log2(3).
    expected = (1 / math.log2(3)) / 1.0
    assert math.isclose(ndcg_at_k([0, 1], k=5), expected)


def test_ndcg_respects_k_cutoff():
    # Relevant item sits at rank 3 but k=2 -> it is excluded -> nDCG 0.
    assert ndcg_at_k([0, 0, 1], k=2) == 0.0


def test_dcg_formula():
    # rel at ranks 1 and 2: 1/log2(2) + 1/log2(3) = 1 + 1/log2(3).
    expected = 1.0 + 1 / math.log2(3)
    assert math.isclose(dcg_at_k([1, 1], k=5), expected)


# --------------------------------------------------------------- score parse
def test_parse_score_plain_float():
    assert parse_score("0.8") == 0.8


def test_parse_score_embedded_in_text():
    assert parse_score("Score: 0.75 out of 1") == 0.75


def test_parse_score_integer():
    assert parse_score("1") == 1.0


def test_parse_score_clamps_out_of_range():
    assert parse_score("1.5") == 1.0
    assert parse_score("-0.2") == 0.0


def test_parse_score_defaults_to_zero_on_garbage():
    assert parse_score("no number here") == 0.0
    assert parse_score("") == 0.0


# ------------------------------------------------ faithfulness judge (mocked)
def test_judge_faithfulness_with_mocked_llm():
    from eval.eval import judge_faithfulness

    class FakeJudge:
        def chat(self, messages, temperature: float = 0.0) -> str:
            return "0.9"

    score = judge_faithfulness(FakeJudge(), "answer", "context")
    assert score == 0.9


def test_judge_faithfulness_empty_inputs_zero():
    from eval.eval import judge_faithfulness

    class FakeJudge:
        def chat(self, messages, temperature: float = 0.0) -> str:  # pragma: no cover
            return "1.0"

    assert judge_faithfulness(FakeJudge(), "", "context") == 0.0

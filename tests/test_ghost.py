import numpy as np
from glassbox.ghost import (
    haunt_level,
    is_countable_token,
    is_ghost,
    p_chosen_and_margin,
    thresholds_for_sensitivity,
)


def test_uniform_distribution_has_zero_margin():
    p, margin = p_chosen_and_margin(np.zeros(4), 1)
    assert abs(p - 0.25) < 1e-9
    assert abs(margin) < 1e-9


def test_confident_distribution():
    p, margin = p_chosen_and_margin(np.array([10.0, 0.0, 0.0, 0.0]), 0)
    assert p > 0.99 and margin > 0.99


def test_ghost_requires_both_uncertainty_signals():
    assert not is_ghost(0.5, 0.95)
    assert is_ghost(5.0, 0.05)
    assert not is_ghost(5.0, 0.5)
    assert not is_ghost(1.0, 0.05)


def test_sensitivity_controls_both_thresholds():
    assert thresholds_for_sensitivity(0) == (6.0, 0.02)
    assert thresholds_for_sensitivity(50) == (4.5, 0.15)
    assert thresholds_for_sensitivity(100) == (3.0, 0.28)


def test_punctuation_and_whitespace_are_not_countable():
    assert is_countable_token("France.")
    assert is_countable_token("éclair")
    assert not is_countable_token("\n")
    assert not is_countable_token("!?")
    assert not is_ghost(8.0, 0.01, token_str="\n")


def test_haunt_level():
    assert haunt_level([]) == 0
    steps = [
        {"entropy_bits": 0.5, "p_chosen": 0.9, "token_str": "solid"} for _ in range(3)
    ] + [
        {"entropy_bits": 5.0, "p_chosen": 0.05, "token_str": "uncertain"},
        {"entropy_bits": 9.0, "p_chosen": 0.01, "token_str": "\n"},
        {"entropy_bits": 9.0, "p_chosen": 0.01, "token_str": "."},
    ]
    assert haunt_level(steps) == 25


def test_sensitivity_reflags_answer_with_same_rule():
    steps = [{"entropy_bits": 5.0, "p_chosen": 0.05, "token_str": "answer"}]
    assert haunt_level(steps, sensitivity=0) == 0
    assert haunt_level(steps, sensitivity=100) == 100
import numpy as np

from glassbox.ghost import p_chosen_and_margin
from glassbox.trace_payload import build_step_payload


class TokenizerStub:
    def decode(self, token_id: int) -> str:
        return f"token-{token_id}"


def test_payload_reports_full_vocabulary_entropy_and_decodes_top_200():
    logits = np.linspace(-4, 4, 256, dtype=np.float32)
    shifted = logits.astype(np.float64)
    shifted -= shifted.max()
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    nonzero = probabilities[probabilities > 0]
    expected_entropy = round(float(-(nonzero * np.log2(nonzero)).sum()), 3)

    payload = build_step_payload(
        step_idx=0,
        token_id=255,
        token_str=" token",
        tokens_per_sec=1.0,
        cache_active=True,
        logits=logits,
        traces=[],
        tokenizer=TokenizerStub(),
        model=None,
        top_n=200,
        include_attention=False,
        include_lens=False,
    )

    assert payload["entropy_bits"] == expected_entropy
    assert len(payload["top_logit_strs"]) == 200
    expected_probability, expected_margin = p_chosen_and_margin(logits, 255)
    assert payload["p_chosen"] == round(expected_probability, 6)
    assert payload["margin"] == round(expected_margin, 4)


def test_probability_signals_do_not_mutate_float64_logits():
    logits = np.array([3.0, 1.0, -2.0], dtype=np.float64)
    original = logits.copy()

    p_chosen_and_margin(logits, 0)

    np.testing.assert_array_equal(logits, original)


def test_payload_does_not_mutate_float64_logits():
    logits = np.array([3.0, 1.0, -2.0], dtype=np.float64)
    original = logits.copy()
    build_step_payload(
        step_idx=0,
        token_id=0,
        token_str="token-0",
        tokens_per_sec=1.0,
        cache_active=True,
        logits=logits,
        traces=[],
        tokenizer=TokenizerStub(),
        model=None,
        top_n=3,
        include_attention=False,
        include_lens=False,
    )
    np.testing.assert_array_equal(logits, original)
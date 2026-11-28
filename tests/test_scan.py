from dataclasses import dataclass

import numpy as np
import pytest

from glassbox.scan import SCAN_PREFIX, scan_text


@dataclass
class FakeOutput:
    logits: np.ndarray


class FakeModel:
    def __init__(self, logits):
        self.logits = np.asarray(logits, dtype=np.float64)
        self.calls = []

    def forward(self, input_ids, cache=None, trace=False):
        self.calls.append((input_ids.copy(), cache, trace))
        logits = np.tile(self.logits, (1, input_ids.shape[1], 1))
        return FakeOutput(logits)


class FakeTokenizer:
    def __init__(self):
        self.add_special_tokens_values = []

    def encode(self, text, add_special_tokens=True):
        self.add_special_tokens_values.append(add_special_tokens)
        if text == SCAN_PREFIX:
            return [0]
        if text == "observed":
            return [1]
        if text == "long text":
            return ([1, 2, 3, 4] * 41)[:161]
        return [1, 2, 3, 4]

    def decode(self, token_ids):
        return f"token-{token_ids}"


def test_scan_scores_observed_token_and_returns_top_three_alternatives():
    model = FakeModel([np.log(2), 0.0, np.log(3), np.log(1.5), np.log(0.5)])
    tokenizer = FakeTokenizer()

    result = scan_text("observed", model, tokenizer)

    record = result["tokens"][0]
    assert record["token_id"] == 1
    assert record["p_chosen"] == pytest.approx(1 / 8)
    assert record["surprise_bits"] == pytest.approx(3.0)
    assert record["margin"] == pytest.approx(1 / 8)
    assert [item["token_str"] for item in record["alternatives"]] == [
        "token-2", "token-0", "token-3",
    ]
    assert len(record["alternatives"]) == 3
    assert len(model.calls) == 1
    np.testing.assert_array_equal(model.calls[0][0], [[0, 1]])
    assert model.calls[0][1:] == (None, False)
    assert tokenizer.add_special_tokens_values == [False, False]


def test_scan_trims_at_150_tokens_and_scores_every_retained_token():
    model = FakeModel(np.zeros(5))

    result = scan_text("long text", model, FakeTokenizer())

    assert result["trimmed"] is True
    assert result["token_count"] == 150
    assert len(result["tokens"]) == 150
    assert model.calls[0][0].shape == (1, 151)


@pytest.mark.parametrize("text", ["", "  \n"])
def test_scan_rejects_empty_text(text):
    with pytest.raises(ValueError, match="Enter some text"):
        scan_text(text, FakeModel(np.zeros(5)), FakeTokenizer())

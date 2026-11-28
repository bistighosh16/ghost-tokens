"""Score observed text with one uncached model forward pass."""
from typing import Protocol

import numpy as np


MAX_SCAN_TOKENS = 150
SCAN_PREFIX = "<|im_start|>assistant\n"


class ScanOutput(Protocol):
    logits: np.ndarray


class ScanModel(Protocol):
    def forward(
        self, input_ids: np.ndarray, cache: None = None, trace: bool = False
    ) -> ScanOutput: ...


class ScanTokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...

    def decode(self, token_ids: list[int] | int) -> str: ...


def scan_text(
    text: str,
    model: ScanModel,
    tokenizer: ScanTokenizer,
    max_tokens: int = MAX_SCAN_TOKENS,
) -> dict:
    """Return surprise, distribution metrics, and alternatives for each text token."""
    if not text.strip():
        raise ValueError("Enter some text to scan.")
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1.")

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        raise ValueError("The text did not produce any model tokens.")

    trimmed = len(token_ids) > max_tokens
    token_ids = token_ids[:max_tokens]
    prefix_ids = tokenizer.encode(SCAN_PREFIX, add_special_tokens=False)
    if not prefix_ids:
        raise ValueError("The tokenizer could not encode the scan context.")
    input_ids = np.asarray([prefix_ids + token_ids], dtype=np.int64)
    logits_by_position = model.forward(input_ids, cache=None, trace=False).logits[0]

    records = []
    for index, token_id in enumerate(token_ids):
        logits = np.asarray(logits_by_position[len(prefix_ids) - 1 + index], dtype=np.float64)
        shifted_logits = logits - np.max(logits)
        log_probabilities = shifted_logits - np.log(np.exp(shifted_logits).sum())
        probabilities = np.exp(log_probabilities)

        top_two = np.partition(probabilities, -2)[-2:]
        top_count = min(4, logits.size)
        top_indices = np.argpartition(logits, -top_count)[-top_count:]
        top_indices = top_indices[np.argsort(logits[top_indices])[::-1]]
        alternatives = [
            {
                "token_str": tokenizer.decode(int(candidate_id)),
                "probability": float(probabilities[candidate_id]),
            }
            for candidate_id in top_indices
            if candidate_id != token_id
        ][:3]

        records.append({
            "index": index,
            "token_id": int(token_id),
            "token_str": tokenizer.decode(int(token_id)),
            "p_chosen": float(probabilities[token_id]),
            "surprise_bits": float(-log_probabilities[token_id] / np.log(2)),
            "entropy_bits": float(-np.sum(
                probabilities * log_probabilities / np.log(2)
            )),
            "margin": float(np.max(top_two) - np.min(top_two)),
            "alternatives": alternatives,
        })

    return {
        "tokens": records,
        "token_count": len(records),
        "trimmed": trimmed,
        "max_tokens": max_tokens,
    }

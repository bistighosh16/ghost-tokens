"""
Ghost Tokens - flag uncertain tokens and score a whole answer.
All signals come from the model's raw temperature-1 distribution,
so changing sampling settings never changes which tokens look unsure.
"""
import numpy as np


def p_chosen_and_margin(logits: np.ndarray, token_id: int) -> tuple[float, float]:
    """Return (probability of the chosen token, top-1 minus top-2 probability)."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max()
    p = np.exp(z)
    p /= p.sum()
    top_two = np.partition(p, -2)[-2:]
    margin = float(top_two.max() - top_two.min())
    return float(p[token_id]), margin


def thresholds_for_sensitivity(sensitivity: float = 50) -> tuple[float, float]:
    """Map 0-100 sensitivity to entropy and chosen-probability cutoffs."""
    level = min(100.0, max(0.0, float(sensitivity))) / 100
    return 6.0 - 3.0 * level, 0.02 + 0.26 * level


def is_countable_token(token_str: str) -> bool:
    """Ignore whitespace and punctuation-only tokens when scoring answers."""
    return any(character.isalnum() for character in token_str)


def is_ghost(entropy_bits: float, p_chosen: float,
             entropy_threshold: float = 4.5, p_threshold: float = 0.15,
             token_str: str | None = None) -> bool:
    """A content token is a ghost only when both uncertainty signals are weak."""
    if token_str is not None and not is_countable_token(token_str):
        return False
    return entropy_bits >= entropy_threshold and p_chosen < p_threshold


def haunt_level(steps: list[dict], *, sensitivity: float = 50,
                entropy_threshold: float | None = None,
                p_threshold: float | None = None) -> int:
    """0-100: percentage of countable answer tokens flagged as ghosts."""
    if entropy_threshold is None or p_threshold is None:
        default_entropy, default_probability = thresholds_for_sensitivity(sensitivity)
        entropy_threshold = default_entropy if entropy_threshold is None else entropy_threshold
        p_threshold = default_probability if p_threshold is None else p_threshold
    answer_steps = [step for step in steps if is_countable_token(step.get("token_str", ""))]
    if not answer_steps:
        return 0
    ghosts = sum(
        1 for s in answer_steps
        if is_ghost(
            s["entropy_bits"], s["p_chosen"],
            entropy_threshold=entropy_threshold,
            p_threshold=p_threshold,
            token_str=s["token_str"],
        )
    )
    return round(100 * ghosts / len(answer_steps))
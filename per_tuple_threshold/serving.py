"""Per-tuple semantic filter used at serving time.

For a new query q' and each tuple d:

    reuse a_d          if s(q, q') >= t_safe(d)
    call LLM(q', d)    otherwise

with ``t_safe(d) = clip(t_hat(d) + margin, 0, 1)``. A tuple without a threshold
(``None``) is never reused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


def should_reuse(similarity: float, threshold: float | None) -> bool:
    return threshold is not None and similarity >= threshold


def safe_threshold(predicted: float | None, margin: float) -> float | None:
    if predicted is None:
        return None
    return float(min(max(predicted + margin, 0.0), 1.0))


@dataclass(frozen=True)
class ReuseDecision:
    reuse: list[str]
    recompute: list[str]


class PerTupleSemanticFilter:
    """Holds t_hat(d) for the tuples cached under one base query q."""

    def __init__(self, thresholds: Mapping[str, float | None], margin: float = 0.0):
        self.margin = margin
        self.thresholds = {
            tuple_id: safe_threshold(value, margin) for tuple_id, value in thresholds.items()
        }

    def decide(self, similarity: float) -> ReuseDecision:
        reuse, recompute = [], []
        for tuple_id, threshold in self.thresholds.items():
            (reuse if should_reuse(similarity, threshold) else recompute).append(tuple_id)
        return ReuseDecision(reuse=reuse, recompute=recompute)

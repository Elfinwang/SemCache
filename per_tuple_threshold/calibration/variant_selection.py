"""Pick calibration and held-out variants at controlled similarity levels.

Each target is matched to the unused candidate of the given category whose
cosine to q is closest (ties go to the higher cosine). The generator never sees
these targets. Unrelated queries are taken from the lowest-cosine end.
"""

from __future__ import annotations

import numpy as np

from per_tuple_threshold.embedding import cosine
from per_tuple_threshold.schema import ROLE_EVAL, ROLE_FIT, QueryVariant

FIT_TARGETS: dict[str, tuple[float, ...]] = {
    "paraphrase": (0.93, 0.89, 0.85, 0.81, 0.77, 0.73, 0.69, 0.65),
    "intensity": (0.78, 0.66, 0.54),
    "should_reuse": (0.84, 0.72),
    "should_not_reuse": (0.58, 0.44, 0.32),
}
EVAL_TARGETS: dict[str, tuple[float, ...]] = {
    "paraphrase": (0.91, 0.83, 0.75, 0.67),
}
UNRELATED_FIT = 4
UNRELATED_EVAL = 2

SIMILARITY_DECIMALS = 4

Scored = list[tuple[str, float]]


def _closest(candidates: Scored, target: float, used: set[str]) -> tuple[str, float] | None:
    remaining = [item for item in candidates if item[0] not in used]
    if not remaining:
        return None
    return min(remaining, key=lambda item: (abs(item[1] - target), -item[1]))


def _lowest(candidates: Scored, count: int, used: set[str]) -> Scored:
    remaining = sorted((item for item in candidates if item[0] not in used), key=lambda item: item[1])
    return remaining[:count]


def select_variants(
    pool: dict[str, list[str]],
    vectors: dict[str, np.ndarray],
    base: str,
) -> list[QueryVariant]:
    """Return the chosen q' in selection order. The base query is not included."""
    base_vector = vectors[base]
    scored: dict[str, Scored] = {
        category: [
            (text, round(cosine(base_vector, vectors[text]), SIMILARITY_DECIMALS))
            for text in rows
        ]
        for category, rows in pool.items()
    }
    used: set[str] = set()
    chosen: list[QueryVariant] = []

    def add(category: str, role: str, hit: tuple[str, float], target: float | None) -> None:
        text, similarity = hit
        used.add(text)
        chosen.append(QueryVariant(
            name=f"{category}_{len(chosen) + 1}",
            role=role,
            category=category,
            text=text,
            similarity=similarity,
            target_similarity=target,
        ))

    def add_targets(targets: dict[str, tuple[float, ...]], role: str) -> None:
        for category, values in targets.items():
            for target in values:
                hit = _closest(scored.get(category, []), target, used)
                if hit is not None:
                    add(category, role, hit, target)

    add_targets(FIT_TARGETS, ROLE_FIT)
    for hit in _lowest(scored.get("unrelated", []), UNRELATED_FIT, used):
        add("unrelated", ROLE_FIT, hit, None)
    add_targets(EVAL_TARGETS, ROLE_EVAL)
    for hit in _lowest(scored.get("unrelated", []), UNRELATED_EVAL, used):
        add("unrelated", ROLE_EVAL, hit, None)
    return chosen

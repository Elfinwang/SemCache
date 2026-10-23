"""Score reuse rules on held-out (``eval``) observations.

A decision is one (q', d) pair. Reusing is correct when LLM(q', d) == a_d;
not reusing is correct when it differs. Held-out queries never enter a fit.
"""

from __future__ import annotations

from typing import Callable, Iterable

from per_tuple_threshold.schema import Observation, TupleCalibration
from per_tuple_threshold.serving import should_reuse

HELD_OUT_GROUPS = ("paraphrase", "unrelated", "all")

ThresholdOf = Callable[[TupleCalibration], float | None]


def score_decisions(rows: Iterable[tuple[Observation, float | None]]) -> dict:
    """Score ``(observation, threshold)`` pairs."""
    decisions = reuses = correct_reuses = correct_rejects = 0
    for obs, threshold in rows:
        decisions += 1
        if should_reuse(obs.similarity, threshold):
            reuses += 1
            correct_reuses += int(obs.match)
        else:
            correct_rejects += int(not obs.match)
    return {
        "decisions": decisions,
        "reuses": reuses,
        "correct_reuses": correct_reuses,
        "false_reuses": reuses - correct_reuses,
        "correct_rejects": correct_rejects,
        "precision": correct_reuses / reuses if reuses else None,
        "decision_accuracy": (correct_reuses + correct_rejects) / decisions if decisions else None,
    }


def score_rule(records: Iterable[TupleCalibration], threshold_of: ThresholdOf) -> dict[str, dict]:
    """Score one rule on held-out observations, per category group."""
    rows: dict[str, list[tuple[Observation, float | None]]] = {name: [] for name in HELD_OUT_GROUPS}
    for record in records:
        threshold = threshold_of(record)
        for obs in record.eval_observations():
            rows["all"].append((obs, threshold))
            if obs.category in rows:
                rows[obs.category].append((obs, threshold))
    return {name: score_decisions(group) for name, group in rows.items()}


def compare_rules(
    records: list[TupleCalibration],
    rules: dict[str, ThresholdOf],
) -> dict[str, dict[str, dict]]:
    """Return ``{group: {rule: score}}``."""
    by_rule = {name: score_rule(records, rule) for name, rule in rules.items()}
    return {group: {name: by_rule[name][group] for name in rules} for group in HELD_OUT_GROUPS}


def sum_scores(scores: Iterable[dict]) -> dict:
    keys = ("decisions", "reuses", "correct_reuses", "false_reuses", "correct_rejects")
    total = {key: 0 for key in keys}
    for score in scores:
        for key in keys:
            total[key] += score[key]
    total["precision"] = total["correct_reuses"] / total["reuses"] if total["reuses"] else None
    total["decision_accuracy"] = (
        (total["correct_reuses"] + total["correct_rejects"]) / total["decisions"]
        if total["decisions"] else None
    )
    return total


def format_score(score: dict) -> str:
    """``reuses/decisions (false) precision`` for terminal tables."""
    if not score["decisions"]:
        return "-"
    precision = "-" if score["precision"] is None else f"{score['precision'] * 100:.1f}%"
    return f"{score['reuses']}/{score['decisions']} ({score['false_reuses']}) {precision}"

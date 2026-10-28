"""Per-clarity breakdown of calibrated thresholds t(d).

Every tuple falls in one bucket, decided by the estimator's own ``reason``
(``calibration/threshold.py``):

* ``fitted``: a threshold was estimated.
* ``all_match``: one label class. The base query always matches, so every
  q' matched: reuse is always safe, i.e. t = 0. Excluded from ``predict``
  training unless ``--all-match-as-zero`` is set.
* ``fit_failed``: logistic regression failed.
* ``too_few``: fewer than 6 observations (does not occur with full pools).

The expected pattern is mean t(d) rising as clarity goes high -> medium -> low.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from per_tuple_threshold.schema import TupleCalibration

BUCKETS = ("fitted", "all_match", "fit_failed", "too_few")
CLARITY_ORDER = ("high", "medium", "low")
TRUTH_ORDER = ("positive", "negative")


def bucket(record: TupleCalibration) -> str:
    if record.threshold.t_hat is not None:
        return "fitted"
    reason = record.threshold.reason
    if "single label class" in reason:
        return "all_match"
    if "fewer than" in reason:
        return "too_few"
    return "fit_failed"


def effective_threshold(record: TupleCalibration) -> float | None:
    """``t_hat`` for fitted rows, 0 for all-match rows, else ``None``."""
    kind = bucket(record)
    if kind == "fitted":
        return record.threshold.t_hat
    return 0.0 if kind == "all_match" else None


def _summarize(records: list[TupleCalibration]) -> dict:
    counts = {name: 0 for name in BUCKETS}
    for record in records:
        counts[bucket(record)] += 1
    fitted = [record.threshold.t_hat for record in records if bucket(record) == "fitted"]
    effective = [value for value in map(effective_threshold, records) if value is not None]
    return {
        "n": len(records),
        **counts,
        "mean_t_fitted": float(np.mean(fitted)) if fitted else None,
        "median_t_fitted": float(np.median(fitted)) if fitted else None,
        "mean_t_effective": float(np.mean(effective)) if effective else None,
    }


def clarity_breakdown(
    records: Iterable[TupleCalibration],
    design: Mapping[str, Mapping[str, str]],
) -> dict:
    """``{"by_clarity": …, "by_truth": …, "by_truth_clarity": …, "all": …}``."""
    records = list(records)

    def group(keys) -> dict:
        groups: dict[str, list[TupleCalibration]] = {}
        for record in records:
            info = design.get(record.tuple_id, {})
            groups.setdefault("/".join(info.get(key, "?") for key in keys), []).append(record)
        return {name: _summarize(rows) for name, rows in sorted(groups.items())}

    by_clarity = group(["clarity"])
    return {
        "by_clarity": {name: by_clarity[name] for name in CLARITY_ORDER if name in by_clarity},
        "by_truth": group(["truth"]),
        "by_truth_clarity": group(["truth", "clarity"]),
        "all": _summarize(records),
    }


def format_breakdown(breakdown: dict) -> str:
    header = f"{'clarity':<10} {'n':>3} {'fitted':>6} {'all_match':>9} {'failed':>6}  {'mean t (fitted)':>15}  {'mean t (all_match=0)':>20}"
    lines = [header, "-" * len(header)]
    for name, row in [*breakdown["by_clarity"].items(), ("all", breakdown["all"])]:
        fitted = "-" if row["mean_t_fitted"] is None else f"{row['mean_t_fitted']:.3f}"
        effective = "-" if row["mean_t_effective"] is None else f"{row['mean_t_effective']:.3f}"
        lines.append(
            f"{name:<10} {row['n']:>3} {row['fitted']:>6} {row['all_match']:>9} "
            f"{row['fit_failed'] + row['too_few']:>6}  {fitted:>15}  {effective:>20}"
        )
    return "\n".join(lines)

"""Conservative additive margin m for predicted thresholds."""

from __future__ import annotations

import numpy as np

MAX_MARGIN = 0.2


def safety_margin(true_thresholds: np.ndarray, predicted: np.ndarray, delta: float = 0.1) -> float:
    """(1 - delta) quantile of ``true - predicted``, clipped to [0, MAX_MARGIN].

    Under-predicting t(d) is the unsafe direction (it reuses too eagerly), so
    the margin covers that side of the residual distribution. ``predicted``
    should be out-of-sample, e.g. leave-one-out predictions.
    """
    residual = np.asarray(true_thresholds, dtype=float) - np.asarray(predicted, dtype=float)
    margin = float(np.quantile(residual, 1.0 - delta))
    return float(np.clip(margin, 0.0, MAX_MARGIN))

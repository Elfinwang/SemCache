"""Per-tuple threshold t(d) from vCache's original estimator.

This module does not re-implement the fit. It calls
``vcache.vcache_policy.strategies.verified._Algorithm._estimate_parameters``,
the function ``VerifiedDecisionPolicy`` uses for every cached embedding:

    P(match | s) = sigmoid(a + gamma * s)      # unpenalised logistic regression
    gamma        = max(gamma, 1e-6)
    t_hat        = clip(-a / gamma, 0, 1)       # rounded to 3 decimals

The guards mirror ``_Algorithm.select_action``: fewer than six observations,
or a failed fit, means the policy keeps exploring and never reuses. A
single-class label set is a failed fit in vCache (scikit-learn raises), so it
is rejected up front to avoid the stack trace vCache prints in that case.

``var_t`` is returned for completeness. The online vCache policy turns it
into a confidence bound; the per-tuple reuse rule here uses ``t_hat`` only.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from per_tuple_threshold.schema import ThresholdEstimate
from vcache.vcache_policy.strategies.verified import _Algorithm

MIN_OBSERVATIONS = 6

# delta only enters vCache's later tau computation, not _estimate_parameters.
_ESTIMATOR = _Algorithm(delta=0.01)


def estimate_threshold(similarities: Sequence[float], labels: Sequence[int]) -> ThresholdEstimate:
    sims = np.asarray(similarities, dtype=float)
    y = np.asarray(labels, dtype=int)
    if len(sims) < MIN_OBSERVATIONS:
        return ThresholdEstimate(None, None, None, "explore: fewer than 6 observations")
    if len(np.unique(y)) < 2:
        return ThresholdEstimate(None, None, None, "explore: single label class")
    t_hat, gamma, var_t = _ESTIMATOR._estimate_parameters(sims, y)
    if t_hat == -1:
        return ThresholdEstimate(None, None, None, "explore: logistic regression failed")
    var_t = float(var_t)
    return ThresholdEstimate(
        t_hat=float(t_hat),
        gamma=float(gamma),
        var_t=var_t if np.isfinite(var_t) else None,
        reason="ok",
    )

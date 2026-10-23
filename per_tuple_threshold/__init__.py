"""Per-tuple similarity thresholds t(d) for vCache-style semantic caching.

A cached answer a_d = LLM(q, d) is reused for a new query q' when
s(q, q') >= t(d). Thresholds are calibrated on a small tuple set with
LLM-generated query variants and vCache's original estimator, then predicted
for the remaining tuples by a small network g_theta.
"""

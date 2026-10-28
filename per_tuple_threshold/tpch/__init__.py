"""Calibrate and predict per-tuple thresholds on the generated TPC-H reviews.

    python -m per_tuple_threshold.tpch calibrate --tpch-dir DIR --output RUN [options]
    python -m per_tuple_threshold.tpch predict --run-dir RUN [options]

This package only imports the core (calibration, labeling, evaluation,
prediction). The core never imports it, and nothing in it is SemBench-specific:
the reviews come from ``data/tpch_reviews`` and the run directory has the same
layout as a SemBench run.
"""

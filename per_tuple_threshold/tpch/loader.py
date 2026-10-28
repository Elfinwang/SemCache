"""TPC-H reviews as ``DataTuple`` populations, plus their design columns.

Reads the generator's output directory (``data/tpch_reviews/generate.py``):
``tpch_sample_ids.json`` lists the frozen sample per population (consistent
reviews only) and ``reviews.parquet`` holds the text and the design columns.
The design columns (``truth``, ``clarity``, ``stratum``) never enter prompts
or features; they are returned separately for analysis.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from per_tuple_threshold.schema import DataTuple

REVIEWS = "reviews.parquet"
SAMPLE_IDS = "tpch_sample_ids.json"
DESIGN_COLUMNS = ("truth", "clarity", "stratum")


def _read(tpch_dir: Path) -> tuple[pd.DataFrame, dict]:
    tpch_dir = Path(tpch_dir)
    for name in (REVIEWS, SAMPLE_IDS):
        if not (tpch_dir / name).is_file():
            raise FileNotFoundError(
                f"{tpch_dir / name} not found; run per_tuple_threshold.data.tpch_reviews.generate first."
            )
    reviews = pd.read_parquet(tpch_dir / REVIEWS).set_index("tuple_id")
    sample = json.loads((tpch_dir / SAMPLE_IDS).read_text())
    return reviews, sample


def load_tpch_populations(tpch_dir: Path) -> dict[str, list[DataTuple]]:
    """One ``DataTuple`` per sample entry, in sample order."""
    reviews, sample = _read(tpch_dir)
    populations = {}
    for name, rows in sample.items():
        missing = [row["tuple_id"] for row in rows if row["tuple_id"] not in reviews.index]
        if missing:
            raise KeyError(f"{len(missing)} sample ids missing from {REVIEWS}, e.g. {missing[:3]}")
        populations[name] = [
            DataTuple(
                tuple_id=row["tuple_id"],
                summary=row["summary"],
                context=f"Review: {reviews.at[row['tuple_id'], 'text']}",
                text=str(reviews.at[row["tuple_id"], "text"]),
            )
            for row in rows
        ]
    return populations


def load_design(tpch_dir: Path) -> dict[str, dict[str, str]]:
    """``tuple_id -> {truth, clarity, stratum}`` for analysis only."""
    reviews, _ = _read(tpch_dir)
    return reviews[list(DESIGN_COLUMNS)].astype(str).to_dict("index")

"""Shared loading for the exploration scripts. Run them from the SemCache repo root with .venv/bin/python."""

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RUN = HERE.parent / "run"
DATA = HERE.parent / "data"
PREDICATE = "tpch_positive"


def load_records(run: Path = RUN) -> list[dict]:
    return json.loads((run / "calibration" / f"{PREDICATE}.json").read_text())["records"]


def load_reviews(data: Path = DATA) -> pd.DataFrame:
    return pd.read_parquet(data / "reviews.parquet").set_index("tuple_id")

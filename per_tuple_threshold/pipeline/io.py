"""Run-directory layout and JSON serialisation.

    <run_dir>/
        run_config.json          settings of the calibrate stage
        generated_pools.json     LLM-generated q' candidates per predicate
        llm_cache.json           cached LLM(q', d) labels
        calibration/<id>.json    per-tuple t(d), observations, held-out scores
        calibration/metrics.json summary of all predicates (no per-tuple records)
        prediction/<id>.json     g_theta predictions, margin, held-out scores
        prediction/metrics.json  summary of all predicates
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from per_tuple_threshold.schema import (
    PredicateCalibration,
    Predicate,
    QueryVariant,
    ThresholdEstimate,
    TupleCalibration,
)

RUN_CONFIG = "run_config.json"
POOLS = "generated_pools.json"
LABEL_CACHE = "llm_cache.json"
CALIBRATION_DIR = "calibration"
PREDICTION_DIR = "prediction"
METRICS = "metrics.json"


def timestamped_dir(root: Path, prefix: str) -> Path:
    return Path(root) / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serialisable: {type(value).__name__}")


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=to_jsonable))


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def load_calibration(path: Path) -> PredicateCalibration:
    payload = read_json(path)
    predicate = payload["predicate"]
    return PredicateCalibration(
        predicate=Predicate(**{**predicate, "covers": tuple(predicate["covers"])}),
        variants=[QueryVariant(**item) for item in payload["variants"]],
        records=[TupleCalibration.from_dict(item) for item in payload["records"]],
        global_threshold=ThresholdEstimate.from_dict(payload["global_threshold"]),
        held_out=payload["held_out"],
    )

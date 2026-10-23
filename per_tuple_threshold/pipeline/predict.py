"""Stage 2: learn g_theta on a calibration subset and predict t(d) elsewhere.

Reads the output of stage 1. Per predicate, the tuples are split into a
calibration set C (|C| = N) and target tuples. g_theta is trained on phi(d)
-> t(d) over C, the margin m is the (1 - delta) quantile of leave-one-out
residuals on C, and target tuples receive t_safe(d) = clip(g_theta(phi(d)) + m).

Target tuples were also calibrated in stage 1. Their t(d) ("oracle") and
held-out labels are used for evaluation only; a deployment would not pay for
them. Rules compared on the held-out variants of the target tuples:

* ``predicted``: t_safe(d)
* ``predicted_no_margin``: g_theta(phi(d))
* ``oracle``: t(d) fitted from the tuple's own LLM labels
* ``global``: t(q) pooled over the calibration set C only
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from per_tuple_threshold.calibration import pooled_threshold
from per_tuple_threshold.embedding import Embedder
from per_tuple_threshold.evaluation import compare_rules, format_score
from per_tuple_threshold.pipeline import io
from per_tuple_threshold.predicates import select_predicates
from per_tuple_threshold.prediction import (
    FEATURE_NAMES,
    FeatureExtractor,
    ThresholdPredictor,
    TrainingConfig,
    safety_margin,
)
from per_tuple_threshold.schema import ROLE_FIT, PredicateCalibration, TupleCalibration
from per_tuple_threshold.serving import safe_threshold

logger = logging.getLogger(__name__)

MIN_TRAINING_TUPLES = 3


@dataclass(frozen=True)
class PredictionConfig:
    run_dir: Path
    embedding_model: str
    calibration_fraction: float = 2 / 3
    margin_delta: float = 0.1
    seed: int = 0
    predicate_ids: tuple[str, ...] | None = None
    training: TrainingConfig = field(default_factory=TrainingConfig)


def run_prediction(config: PredictionConfig) -> Path:
    run_dir = Path(config.run_dir)
    calibration_dir = run_dir / io.CALIBRATION_DIR
    if not calibration_dir.is_dir():
        raise FileNotFoundError(f"No calibration results in {run_dir}; run the calibrate stage first.")
    predicates = select_predicates(list(config.predicate_ids) if config.predicate_ids else None)
    extractor = FeatureExtractor(Embedder(config.embedding_model))

    summaries = []
    for predicate in predicates:
        path = calibration_dir / f"{predicate.id}.json"
        if not path.is_file():
            logger.warning("Skipping %s: %s not found", predicate.id, path)
            continue
        result = predict_predicate(io.load_calibration(path), extractor, config)
        io.write_json(run_dir / io.PREDICTION_DIR / f"{predicate.id}.json", result)
        summaries.append({key: value for key, value in result.items() if key != "tuples"})

    io.write_json(run_dir / io.PREDICTION_DIR / io.METRICS, {
        "config": asdict(config),
        "feature_names": list(FEATURE_NAMES),
        "predicates": summaries,
    })
    _log_table(summaries)
    return run_dir


def split_tuples(
    records: list[TupleCalibration],
    fraction: float,
    seed: int,
) -> tuple[list[TupleCalibration], list[TupleCalibration]]:
    """Split by unique tuple id so a duplicated tuple lands on one side only."""
    ids = list(dict.fromkeys(record.tuple_id for record in records))
    count = min(max(int(round(fraction * len(ids))), 1), len(ids) - 1)
    order = np.random.default_rng(seed).permutation(len(ids))
    chosen = {ids[index] for index in order[:count]}
    calibration = [record for record in records if record.tuple_id in chosen]
    target = [record for record in records if record.tuple_id not in chosen]
    return calibration, target


def predict_predicate(
    calibration_result: PredicateCalibration,
    extractor: FeatureExtractor,
    config: PredictionConfig,
) -> dict:
    predicate = calibration_result.predicate
    records = calibration_result.records
    calibration, target = split_tuples(records, config.calibration_fraction, config.seed)
    training = [record for record in calibration if record.threshold.t_hat is not None]
    n_fit_queries = sum(item.role == ROLE_FIT for item in calibration_result.variants)
    summary = {
        "id": predicate.id,
        "kind": predicate.kind,
        "population": predicate.population,
        "cost": _cost(records, calibration, n_fit_queries),
        "n_training_tuples": len(training),
    }
    if len(training) < MIN_TRAINING_TUPLES:
        logger.warning("Skipping %s: only %d calibration tuples have t(d)", predicate.id, len(training))
        return {**summary, "status": "skipped: too few calibration thresholds", "tuples": []}

    kind = predicate.kind
    train_x = extractor.transform([r.text for r in training], [r.cached_answer for r in training], kind)
    train_y = np.asarray([r.threshold.t_hat for r in training], dtype=np.float32)
    target_x = extractor.transform([r.text for r in target], [r.cached_answer for r in target], kind)

    predictor = ThresholdPredictor(train_x.shape[1], config.training)
    history = predictor.fit(train_x, train_y)
    loo = predictor.leave_one_out(train_x, train_y)
    margin = safety_margin(train_y, loo, delta=config.margin_delta)
    predicted = dict(zip((r.tuple_id for r in target), predictor.predict(target_x).tolist()))

    global_t = pooled_threshold(calibration)
    held_out = compare_rules(target, {
        "predicted": lambda r: safe_threshold(predicted[r.tuple_id], margin),
        "predicted_no_margin": lambda r: predicted[r.tuple_id],
        "oracle": lambda r: r.threshold.t_hat,
        "global": lambda r: global_t.t_hat,
    })
    errors = [
        abs(predicted[r.tuple_id] - r.threshold.t_hat)
        for r in target
        if r.threshold.t_hat is not None
    ]
    tuples = [
        {
            "tuple_id": r.tuple_id,
            "split": "calibration",
            "t_hat": r.threshold.t_hat,
            "features": features.tolist(),
            "loo_prediction": float(loo_value),
        }
        for r, features, loo_value in zip(training, train_x, loo)
    ] + [
        {
            "tuple_id": r.tuple_id,
            "split": "target",
            "t_hat": r.threshold.t_hat,
            "features": features.tolist(),
            "predicted": predicted[r.tuple_id],
            "safe_threshold": safe_threshold(predicted[r.tuple_id], margin),
        }
        for r, features in zip(target, target_x)
    ]
    return {
        **summary,
        "status": "ok",
        "margin": margin,
        "final_train_loss": history[-1],
        "loo_mae": float(np.mean(np.abs(loo - train_y))),
        "target_mae": float(np.mean(errors)) if errors else None,
        "global_threshold": asdict(global_t),
        "held_out": held_out,
        "tuples": tuples,
    }


def _cost(records: list[TupleCalibration], calibration: list[TupleCalibration], k: int) -> dict:
    """LLM calls of a deployment: M base answers plus N * K calibration labels."""
    m = len({record.tuple_id for record in records})
    n = len({record.tuple_id for record in calibration})
    return {"M": m, "N": n, "K": k, "llm_calls": m + n * k, "per_tuple_calibration_calls": m * (k + 1)}


def _log_table(summaries: list[dict]) -> None:
    header = (
        f"{'predicate':<30} {'N/M':>6} {'margin':>7} {'LOO MAE':>8}  "
        f"{'predicted':<22} {'oracle':<22} {'global':<22}"
    )
    lines = [header, "-" * len(header)]
    for row in summaries:
        cost = f"{row['cost']['N']}/{row['cost']['M']}"
        if row["status"] != "ok":
            lines.append(f"{row['id']:<30} {cost:>6}  {row['status']}")
            continue
        scores = row["held_out"]["all"]
        lines.append(
            f"{row['id']:<30} {cost:>6} {row['margin']:>7.3f} {row['loo_mae']:>8.3f}  "
            f"{format_score(scores['predicted']):<22} {format_score(scores['oracle']):<22} "
            f"{format_score(scores['global']):<22}"
        )
    logger.info("Prediction results on target tuples, reuses/decisions (false reuses) precision\n%s", "\n".join(lines))

"""Stages 1 and 2 on TPC-H reviews, composed from the core's public steps.

Stage 1 (``run_tpch_calibration``) follows ``pipeline/calibrate.py``:
generate q' pools, embed and select variants, label every tuple, fit t(d)
per tuple and a pooled t(q), score them on held-out variants. It adds
``calibration/clarity.json``, the per-clarity breakdown of t(d).

Stage 2 (``run_tpch_prediction``) calls the core ``predict_predicate`` with
the TPC-H feature extractor and adds the mean predicted threshold per clarity
of the target tuples. With ``all_match_as_zero`` the all-match tuples (one
label class, see ``analysis.py``) enter training and evaluation with t = 0
instead of being dropped; this also changes the ``oracle`` rule.

The run directory has the same layout as a SemBench run. Both stages accept
an injected embedder for offline tests; LLM clients come from ``make_client``.
"""

from __future__ import annotations

import gc
import logging
import shutil
from dataclasses import asdict, dataclass, replace
from itertools import chain
from pathlib import Path

import numpy as np

from per_tuple_threshold.calibration import calibrate_predicate, label_jobs, pooled_threshold, select_variants
from per_tuple_threshold.calibration.calibrator import LabelKey
from per_tuple_threshold.calibration.query_generation import CATEGORIES
from per_tuple_threshold.evaluation import compare_rules, format_score
from per_tuple_threshold.llm import CachedLabeler, label_all, make_client
from per_tuple_threshold.pipeline import io
from per_tuple_threshold.pipeline.calibrate import summarize
from per_tuple_threshold.pipeline.predict import PredictionConfig, predict_predicate
from per_tuple_threshold.schema import (
    DataTuple,
    Predicate,
    PredicateCalibration,
    QueryVariant,
    ThresholdEstimate,
)
from per_tuple_threshold.tpch import analysis
from per_tuple_threshold.tpch.features import FEATURE_NAMES, TpchFeatureExtractor
from per_tuple_threshold.tpch.loader import load_design, load_tpch_populations
from per_tuple_threshold.tpch.predicates import select_predicates
from per_tuple_threshold.tpch.query_generation import generate_pool

logger = logging.getLogger(__name__)

DATASET = "tpch"
CLARITY = "clarity.json"
GENERATOR_MANIFEST = "manifest.json"


@dataclass(frozen=True)
class TpchCalibrationConfig:
    output_dir: Path
    tpch_dir: Path
    embedding_model: str
    llm_provider: str
    llm_model: str
    predicate_ids: tuple[str, ...] | None = None
    pools_path: Path | None = None
    label_cache_path: Path | None = None
    workers: int = 8


def _release_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _make_embedder(model: str, embedder):
    if embedder is not None:
        return embedder, False
    from per_tuple_threshold.embedding import Embedder

    return Embedder(model), True


# --------------------------------------------------------------------------- stage 1


def run_tpch_calibration(config: TpchCalibrationConfig, embedder=None) -> Path:
    predicates = select_predicates(list(config.predicate_ids) if config.predicate_ids else None)
    populations = load_tpch_populations(config.tpch_dir)
    design = load_design(config.tpch_dir)

    run_dir = Path(config.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(config.tpch_dir) / GENERATOR_MANIFEST
    manifest = io.read_json(manifest_path) if manifest_path.is_file() else {}
    io.write_json(run_dir / io.RUN_CONFIG, {
        **asdict(config),
        "dataset": DATASET,
        "generator_prompt_version": manifest.get("prompt_version"),
        "generator_model": manifest.get("generator_model"),
    })
    logger.info("Run directory %s", run_dir)

    pools = _load_or_generate_pools(predicates, config, run_dir)
    variants = _select_all_variants(predicates, pools, config.embedding_model, embedder)

    labeler = CachedLabeler(config.llm_provider, config.llm_model, _prepare_label_cache(config, run_dir))
    jobs = chain.from_iterable(
        label_jobs(predicate, populations[predicate.population], variants[predicate.id])
        for predicate in predicates
    )
    answers = label_all(jobs, labeler, config.workers)

    results, breakdowns = [], {}
    for predicate in predicates:
        result = _calibrate(predicate, populations[predicate.population], variants[predicate.id], answers)
        io.write_json(run_dir / io.CALIBRATION_DIR / f"{predicate.id}.json", result)
        results.append(result)
        breakdowns[predicate.id] = analysis.clarity_breakdown(result.records, design)
        logger.info("t(d) by clarity for %s\n%s", predicate.id, analysis.format_breakdown(breakdowns[predicate.id]))

    io.write_json(run_dir / io.CALIBRATION_DIR / CLARITY, breakdowns)
    io.write_json(run_dir / io.CALIBRATION_DIR / io.METRICS, {
        "dataset": DATASET,
        "llm": {
            "provider": config.llm_provider,
            "model": config.llm_model,
            "label_api_calls": labeler.api_calls,
            "label_cache_entries": len(labeler.cache),
        },
        "predicates": [summarize(result) for result in results],
    })
    _log_calibration_table(results)
    return run_dir


def _load_or_generate_pools(
    predicates: list[Predicate],
    config: TpchCalibrationConfig,
    run_dir: Path,
) -> dict[str, dict[str, list[str]]]:
    if config.pools_path is not None:
        saved = io.read_json(config.pools_path)
        missing = [predicate.id for predicate in predicates if predicate.id not in saved]
        if missing:
            raise KeyError(f"{config.pools_path} has no pool for {missing}")
        pools = {predicate.id: saved[predicate.id] for predicate in predicates}
        logger.info("Loaded q' pools from %s", config.pools_path)
    else:
        client = make_client(config.llm_provider, config.llm_model)
        pools = {}
        for predicate in predicates:
            pools[predicate.id] = generate_pool(predicate, client)
            sizes = {category: len(pools[predicate.id][category]) for category in CATEGORIES}
            logger.info("Generated q' pool for %s %s", predicate.id, sizes)
    io.write_json(run_dir / io.POOLS, pools)
    return pools


def _select_all_variants(
    predicates: list[Predicate],
    pools: dict[str, dict[str, list[str]]],
    embedding_model: str,
    embedder,
) -> dict[str, list[QueryVariant]]:
    texts = [predicate.base for predicate in predicates]
    for predicate in predicates:
        for rows in pools[predicate.id].values():
            texts.extend(rows)
    embedder, owned = _make_embedder(embedding_model, embedder)
    vectors = embedder.encode_queries(texts)
    if owned:
        del embedder
        _release_gpu()
    return {predicate.id: select_variants(pools[predicate.id], vectors, predicate.base) for predicate in predicates}


def _prepare_label_cache(config: TpchCalibrationConfig, run_dir: Path) -> Path:
    """The run always writes its own cache; ``label_cache_path`` only seeds it."""
    target = run_dir / io.LABEL_CACHE
    if config.label_cache_path is not None and not target.exists():
        shutil.copyfile(config.label_cache_path, target)
        logger.info("Seeded label cache from %s", config.label_cache_path)
    return target


def _calibrate(
    predicate: Predicate,
    tuples: list[DataTuple],
    variants: list[QueryVariant],
    answers: dict[LabelKey, str],
) -> PredicateCalibration:
    records = calibrate_predicate(predicate, tuples, variants, answers)
    unified = pooled_threshold(records)
    held_out = compare_rules(records, {
        "per_tuple": lambda record: record.threshold.t_hat,
        "unified": lambda record: unified.t_hat,
    })
    return PredicateCalibration(
        predicate=predicate,
        variants=variants,
        records=records,
        global_threshold=unified,
        held_out=held_out,
    )


def _log_calibration_table(results: list[PredicateCalibration]) -> None:
    header = f"{'predicate':<20} {'n':>3} {'t(d) range':>13} {'t(q)':>6}  {'held-out all: per-tuple':<26} {'unified':<26}"
    lines = [header, "-" * len(header)]
    for result in results:
        row = summarize(result)
        span = "-" if row["t_hat_min"] is None else f"{row['t_hat_min']:.3f}-{row['t_hat_max']:.3f}"
        unified = "-" if result.global_threshold.t_hat is None else f"{result.global_threshold.t_hat:.3f}"
        scores = result.held_out["all"]
        lines.append(
            f"{row['id']:<20} {row['n_tuples']:>3} {span:>13} {unified:>6}  "
            f"{format_score(scores['per_tuple']):<26} {format_score(scores['unified']):<26}"
        )
    logger.info("Calibration results, reuses/decisions (false reuses) precision\n%s", "\n".join(lines))


# --------------------------------------------------------------------------- stage 2


def with_all_match_as_zero(result: PredicateCalibration) -> PredicateCalibration:
    """Copy of ``result`` where all-match tuples carry t = 0 instead of ``None``."""
    zero = ThresholdEstimate(t_hat=0.0, gamma=None, var_t=None, reason="all match: set to 0")
    records = [
        replace(record, threshold=zero) if analysis.bucket(record) == "all_match" else record
        for record in result.records
    ]
    return replace(result, records=records)


def run_tpch_prediction(
    config: PredictionConfig,
    all_match_as_zero: bool = False,
    embedder=None,
) -> Path:
    run_dir = Path(config.run_dir)
    calibration_dir = run_dir / io.CALIBRATION_DIR
    if not calibration_dir.is_dir():
        raise FileNotFoundError(f"No calibration results in {run_dir}; run the calibrate stage first.")
    run_config = io.read_json(run_dir / io.RUN_CONFIG)
    design = load_design(Path(run_config["tpch_dir"]))
    predicates = select_predicates(list(config.predicate_ids) if config.predicate_ids else None)
    embedder, _ = _make_embedder(config.embedding_model, embedder)
    extractor = TpchFeatureExtractor(embedder)

    summaries = []
    for predicate in predicates:
        path = calibration_dir / f"{predicate.id}.json"
        if not path.is_file():
            logger.warning("Skipping %s: %s not found", predicate.id, path)
            continue
        calibration = io.load_calibration(path)
        if all_match_as_zero:
            calibration = with_all_match_as_zero(calibration)
        result = predict_predicate(calibration, extractor, config)
        result["all_match_as_zero"] = all_match_as_zero
        result["predicted_by_clarity"] = _predicted_by_clarity(result.get("tuples", []), design)
        io.write_json(run_dir / io.PREDICTION_DIR / f"{predicate.id}.json", result)
        summaries.append({key: value for key, value in result.items() if key != "tuples"})

    io.write_json(run_dir / io.PREDICTION_DIR / io.METRICS, {
        "dataset": DATASET,
        "config": asdict(config),
        "all_match_as_zero": all_match_as_zero,
        "feature_names": list(FEATURE_NAMES),
        "predicates": summaries,
    })
    _log_prediction_table(summaries)
    return run_dir


def _predicted_by_clarity(tuples: list[dict], design: dict) -> dict:
    """Mean predicted and safe threshold of the target tuples, per clarity."""
    groups: dict[str, list[dict]] = {}
    for row in tuples:
        if row["split"] == "target":
            groups.setdefault(design.get(row["tuple_id"], {}).get("clarity", "?"), []).append(row)
    return {
        clarity: {
            "n": len(rows),
            "mean_predicted": float(np.mean([row["predicted"] for row in rows])),
            "mean_safe_threshold": float(np.mean([row["safe_threshold"] for row in rows])),
        }
        for clarity, rows in sorted(groups.items())
    }


def _log_prediction_table(summaries: list[dict]) -> None:
    header = (
        f"{'predicate':<20} {'N/M':>6} {'margin':>7} {'LOO MAE':>8}  "
        f"{'predicted':<22} {'oracle':<22} {'global':<22}"
    )
    lines = [header, "-" * len(header)]
    for row in summaries:
        cost = f"{row['cost']['N']}/{row['cost']['M']}"
        if row["status"] != "ok":
            lines.append(f"{row['id']:<20} {cost:>6}  {row['status']}")
            continue
        scores = row["held_out"]["all"]
        lines.append(
            f"{row['id']:<20} {cost:>6} {row['margin']:>7.3f} {row['loo_mae']:>8.3f}  "
            f"{format_score(scores['predicted']):<22} {format_score(scores['oracle']):<22} "
            f"{format_score(scores['global']):<22}"
        )
        by_clarity = ", ".join(
            f"{name}: {values['mean_predicted']:.3f} (n={values['n']})"
            for name, values in row["predicted_by_clarity"].items()
        )
        lines.append(f"{'':<20}   mean predicted t by clarity: {by_clarity}")
    logger.info("Prediction results on target tuples, reuses/decisions (false reuses) precision\n%s", "\n".join(lines))

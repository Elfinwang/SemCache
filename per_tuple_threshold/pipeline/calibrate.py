"""Stage 1: estimate t(d) on calibration tuples from LLM-generated q'.

1. Generate a candidate pool of q' per base query with the LLM
   (or load a saved ``generated_pools.json``).
2. Embed q and all candidates; select fit / held-out variants by similarity.
3. Label LLM(q, d) and LLM(q', d) for every tuple and variant.
4. Fit t(d) per tuple and a pooled t(q) with vCache's estimator.
5. Score per-tuple t(d) against the pooled t(q) on held-out variants.
"""

from __future__ import annotations

import gc
import logging
import shutil
from dataclasses import asdict, dataclass
from itertools import chain
from pathlib import Path
from typing import Mapping

from per_tuple_threshold.calibration import (
    calibrate_predicate,
    generate_pool,
    label_jobs,
    pooled_threshold,
    select_variants,
)
from per_tuple_threshold.calibration.calibrator import LabelKey
from per_tuple_threshold.calibration.query_generation import CATEGORIES
from per_tuple_threshold.config import reviews_path
from per_tuple_threshold.data import load_populations, load_reviews
from per_tuple_threshold.embedding import Embedder
from per_tuple_threshold.evaluation import compare_rules, format_score
from per_tuple_threshold.llm import CachedLabeler, label_all, make_client
from per_tuple_threshold.pipeline import io
from per_tuple_threshold.predicates import select_predicates
from per_tuple_threshold.schema import (
    ROLE_EVAL,
    ROLE_FIT,
    DataTuple,
    Predicate,
    PredicateCalibration,
    QueryVariant,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CalibrationConfig:
    output_dir: Path
    sembench_root: Path
    embedding_model: str
    llm_provider: str
    llm_model: str
    predicate_ids: tuple[str, ...] | None = None
    pools_path: Path | None = None
    label_cache_path: Path | None = None
    workers: int = 8


def run_calibration(config: CalibrationConfig) -> Path:
    predicates = select_predicates(list(config.predicate_ids) if config.predicate_ids else None)
    reviews_csv = reviews_path(config.sembench_root)
    if not reviews_csv.is_file():
        raise FileNotFoundError(f"SemBench reviews not found: {reviews_csv}")

    run_dir = Path(config.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    io.write_json(run_dir / io.RUN_CONFIG, {**asdict(config), "reviews_csv": reviews_csv})
    logger.info("Run directory %s", run_dir)

    pools = _load_or_generate_pools(predicates, config, run_dir)
    variants = _select_all_variants(predicates, pools, config.embedding_model)

    populations = load_populations(load_reviews(reviews_csv))
    labeler = CachedLabeler(config.llm_provider, config.llm_model, _prepare_label_cache(config, run_dir))
    jobs = chain.from_iterable(
        label_jobs(predicate, populations[predicate.population], variants[predicate.id])
        for predicate in predicates
    )
    answers = label_all(jobs, labeler, config.workers)

    results = []
    for predicate in predicates:
        result = _calibrate(predicate, populations[predicate.population], variants[predicate.id], answers)
        io.write_json(run_dir / io.CALIBRATION_DIR / f"{predicate.id}.json", result)
        results.append(result)

    io.write_json(run_dir / io.CALIBRATION_DIR / io.METRICS, {
        "llm": {
            "provider": config.llm_provider,
            "model": config.llm_model,
            "label_api_calls": labeler.api_calls,
            "label_cache_entries": len(labeler.cache),
        },
        "predicates": [summarize(result) for result in results],
    })
    _log_table(results)
    return run_dir


def _load_or_generate_pools(
    predicates: list[Predicate],
    config: CalibrationConfig,
    run_dir: Path,
) -> dict[str, dict[str, list[str]]]:
    if config.pools_path is not None:
        # Pools written before the refactor are keyed by "<predicate id>.json".
        saved = {key.removesuffix(".json"): value for key, value in io.read_json(config.pools_path).items()}
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
) -> dict[str, list[QueryVariant]]:
    texts = [predicate.base for predicate in predicates]
    for predicate in predicates:
        for rows in pools[predicate.id].values():
            texts.extend(rows)
    embedder = Embedder(embedding_model)
    vectors = embedder.encode_queries(texts)
    del embedder
    _release_gpu()
    selected = {}
    for predicate in predicates:
        selected[predicate.id] = select_variants(pools[predicate.id], vectors, predicate.base)
        logger.debug(
            "Selected %s: %s",
            predicate.id,
            " ".join(f"{item.category}:{item.similarity:.2f}" for item in selected[predicate.id]),
        )
    return selected


def _prepare_label_cache(config: CalibrationConfig, run_dir: Path) -> Path:
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
    answers: Mapping[LabelKey, str],
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


def summarize(result: PredicateCalibration) -> dict:
    fitted = [record.threshold.t_hat for record in result.records if record.threshold.t_hat is not None]
    return {
        "id": result.predicate.id,
        "kind": result.predicate.kind,
        "population": result.predicate.population,
        "covers": list(result.predicate.covers),
        "base": result.predicate.base,
        "n_tuples": len(result.records),
        "n_fit_queries": 1 + sum(item.role == ROLE_FIT for item in result.variants),
        "n_eval_queries": sum(item.role == ROLE_EVAL for item in result.variants),
        "t_hat_min": min(fitted) if fitted else None,
        "t_hat_max": max(fitted) if fitted else None,
        "n_no_threshold": sum(record.threshold.t_hat is None for record in result.records),
        "n_zero_threshold": sum(record.threshold.t_hat == 0 for record in result.records),
        "global_threshold": asdict(result.global_threshold),
        "held_out": result.held_out,
    }


def _log_table(results: list[PredicateCalibration]) -> None:
    header = f"{'predicate':<30} {'n':>3} {'t(d) range':>13} {'t(q)':>6}  {'held-out all: per-tuple':<26} {'unified':<26}"
    lines = [header, "-" * len(header)]
    for result in results:
        row = summarize(result)
        span = "-" if row["t_hat_min"] is None else f"{row['t_hat_min']:.3f}-{row['t_hat_max']:.3f}"
        unified = "-" if result.global_threshold.t_hat is None else f"{result.global_threshold.t_hat:.3f}"
        scores = result.held_out["all"]
        lines.append(
            f"{row['id']:<30} {row['n_tuples']:>3} {span:>13} {unified:>6}  "
            f"{format_score(scores['per_tuple']):<26} {format_score(scores['unified']):<26}"
        )
    logger.info("Calibration results, reuses/decisions (false reuses) precision\n%s", "\n".join(lines))


def _release_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

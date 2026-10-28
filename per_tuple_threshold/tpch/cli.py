"""Command-line interface for the TPC-H reviews.

    python -m per_tuple_threshold.tpch calibrate --tpch-dir DIR [options]
    python -m per_tuple_threshold.tpch predict --run-dir DIR [options]

Flags match ``python -m per_tuple_threshold`` where they overlap.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from per_tuple_threshold import config
from per_tuple_threshold.embedding import configure_offline_mode
from per_tuple_threshold.llm.client import LOCAL_BASE_URL_ENV, PROVIDERS
from per_tuple_threshold.pipeline import io
from per_tuple_threshold.prediction.model import TrainingConfig
from per_tuple_threshold.tpch.predicates import PREDICATES

TPCH_DIR_ENV = "TPCH_DIR"


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--predicates",
        nargs="+",
        default=None,
        metavar="ID",
        help="Predicate ids to run (default: all). Choices: " + ", ".join(p.id for p in PREDICATES),
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m per_tuple_threshold.tpch",
        description="Per-tuple similarity thresholds on generated TPC-H reviews.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    calibrate = commands.add_parser("calibrate", help="Generate q', label reviews, and fit t(d).")
    calibrate.add_argument(
        "--tpch-dir",
        type=Path,
        default=os.environ.get(TPCH_DIR_ENV),
        required=TPCH_DIR_ENV not in os.environ,
        help=f"Output directory of data/tpch_reviews/generate.py (env {TPCH_DIR_ENV}).",
    )
    calibrate.add_argument(
        "--embedding-model",
        default=config.default_embedding_model(),
        help=f"HF model id or local directory (env {config.EMBEDDING_MODEL_ENV}; default: %(default)s).",
    )
    calibrate.add_argument(
        "--llm-provider",
        choices=sorted(PROVIDERS),
        default=None,
        help=f"Labeling LLM provider (env {config.LLM_PROVIDER_ENV}; default: local if {LOCAL_BASE_URL_ENV} is set).",
    )
    calibrate.add_argument("--llm-model", default=None, help=f"Labeling model (env {config.LLM_MODEL_ENV}).")
    calibrate.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Run directory (default: {config.RESULTS_DIR}/tpch_run_<timestamp>).",
    )
    calibrate.add_argument("--pools", type=Path, default=None, help="Reuse a saved generated_pools.json.")
    calibrate.add_argument("--label-cache", type=Path, default=None, help="Seed the label cache from an llm_cache.json.")
    calibrate.add_argument("--workers", type=int, default=8, help="Parallel LLM requests.")
    _add_common(calibrate)

    predict = commands.add_parser("predict", help="Train g_theta on calibration tuples and predict t(d).")
    predict.add_argument("--run-dir", type=Path, required=True, help="Output directory of the calibrate stage.")
    predict.add_argument(
        "--embedding-model",
        default=None,
        help="HF model id or local directory (default: the one in run_config.json).",
    )
    predict.add_argument("--calibration-fraction", type=float, default=2 / 3, help="Share of tuples in C.")
    predict.add_argument("--margin-delta", type=float, default=0.1, help="Margin = (1 - delta) residual quantile.")
    predict.add_argument("--seed", type=int, default=0)
    predict.add_argument("--hidden-dim", type=int, default=TrainingConfig.hidden_dim)
    predict.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    predict.add_argument("--learning-rate", type=float, default=TrainingConfig.learning_rate)
    predict.add_argument("--weight-decay", type=float, default=TrainingConfig.weight_decay)
    predict.add_argument(
        "--all-match-as-zero",
        action="store_true",
        help="Train and evaluate with t = 0 for all-match tuples instead of dropping them (changes the oracle rule).",
    )
    _add_common(predict)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    predicate_ids = tuple(args.predicates) if args.predicates else None

    embedding_model = args.embedding_model
    if embedding_model is None:
        embedding_model = io.read_json(args.run_dir / io.RUN_CONFIG)["embedding_model"]
    configure_offline_mode(embedding_model)

    if args.command == "calibrate":
        from per_tuple_threshold.tpch.pipeline import TpchCalibrationConfig, run_tpch_calibration

        llm_provider, llm_model = config.resolve_llm(args.llm_model, args.llm_provider)
        run_dir = run_tpch_calibration(TpchCalibrationConfig(
            output_dir=args.output or io.timestamped_dir(config.RESULTS_DIR, "tpch_run"),
            tpch_dir=args.tpch_dir,
            embedding_model=embedding_model,
            llm_provider=llm_provider,
            llm_model=llm_model,
            predicate_ids=predicate_ids,
            pools_path=args.pools,
            label_cache_path=args.label_cache,
            workers=args.workers,
        ))
    else:
        from per_tuple_threshold.pipeline.predict import PredictionConfig
        from per_tuple_threshold.tpch.pipeline import run_tpch_prediction

        run_dir = run_tpch_prediction(
            PredictionConfig(
                run_dir=args.run_dir,
                embedding_model=embedding_model,
                calibration_fraction=args.calibration_fraction,
                margin_delta=args.margin_delta,
                seed=args.seed,
                predicate_ids=predicate_ids,
                training=TrainingConfig(
                    hidden_dim=args.hidden_dim,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    seed=args.seed,
                ),
            ),
            all_match_as_zero=args.all_match_as_zero,
        )
    logging.getLogger(__name__).info("Results written to %s", run_dir)

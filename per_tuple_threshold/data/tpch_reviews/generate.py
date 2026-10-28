"""Build the TPC-H reviews dataset.

    # once, on a machine with internet access (e.g. the Fir login node)
    python -m per_tuple_threshold.data.tpch_reviews.generate --dbgen-only --sf 0.01 --out DIR

    # inspect the sample and design without any LLM call
    python -m per_tuple_threshold.data.tpch_reviews.generate --dry-run --n 30 --out DIR

    # generate reviews (an OpenAI-compatible server such as vLLM must be running)
    python -m per_tuple_threshold.data.tpch_reviews.generate --n 30 --llm-model llama-3.1-8b --out DIR

Steps: facts -> stratified sample -> truth/clarity -> write review -> validate
(retry up to ``MAX_RETRIES`` times) -> ``reviews.parquet``,
``tpch_sample_ids.json`` and ``manifest.json`` in ``--out``. Every LLM answer
is cached in ``gen_cache.json`` keyed by (prompt version, model, kind,
prompt, attempt), so an interrupted run resumes and a retry after a failed
validation asks the model again instead of hitting the cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from per_tuple_threshold import config
from per_tuple_threshold.data.tpch_reviews import facts as tpch_facts
from per_tuple_threshold.data.tpch_reviews import prompts
from per_tuple_threshold.llm.client import ChatClient, make_client
from per_tuple_threshold.pipeline import io

logger = logging.getLogger(__name__)

DATABASE = "tpch.duckdb"
CACHE = "gen_cache.json"
REVIEWS = "reviews.parquet"
SAMPLE_IDS = "tpch_sample_ids.json"
MANIFEST = "manifest.json"
DRY_RUN = "sample_dry_run.parquet"

POPULATION = "tpch_reviews"
MAX_RETRIES = 2
FLUSH_EVERY = 20


class GenerationCache:
    """Thread-safe JSON cache of LLM answers for one generator model."""

    def __init__(self, client: ChatClient, path: Path):
        self.client = client
        self.path = Path(path)
        self.entries: dict[str, str] = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.api_calls = 0
        self._lock = threading.Lock()

    def complete(self, kind: str, prompt: str, attempt: int, max_tokens: int, temperature: float) -> str:
        key = hashlib.sha256(
            f"{prompts.PROMPT_VERSION}\n{self.client.model}\n{kind}\n{attempt}\n{prompt}".encode()
        ).hexdigest()
        with self._lock:
            cached = self.entries.get(key)
        if cached is not None:
            return cached
        text = self.client.complete([{"role": "user", "content": prompt}], max_tokens, temperature)
        with self._lock:
            self.entries[key] = text
            self.api_calls += 1
            if self.api_calls % FLUSH_EVERY == 0:
                self._write()
        return text

    def flush(self) -> None:
        with self._lock:
            self._write()

    def _write(self) -> None:
        self.path.write_text(json.dumps(self.entries))


def review_item(item: dict, cache: GenerationCache) -> dict:
    """Write and validate one review; regenerate with a new attempt number on failure."""
    result: dict = {}
    for attempt in range(MAX_RETRIES + 1):
        text = prompts.clean_review(cache.complete(
            "generate", prompts.generation_prompt(item), attempt,
            prompts.GENERATION_MAX_TOKENS, prompts.GENERATION_TEMPERATURE,
        ))
        answer = cache.complete(
            "validate", prompts.validation_prompt(item, text), attempt,
            prompts.VALIDATION_MAX_TOKENS, prompts.VALIDATION_TEMPERATURE,
        )
        try:
            check = prompts.parse_validation(answer)
        except ValueError as exc:
            logger.debug("%s attempt %d: %s", item["tuple_id"], attempt, exc)
            result = {"text": text, "consistent": False, "check_contradiction": None,
                      "check_states_rating": None, "check_sentiment": None, "attempts": attempt + 1}
            continue
        result = {
            "text": text,
            "consistent": check.passes(item["truth"], item["clarity"]),
            "check_contradiction": check.contradicts,
            "check_states_rating": check.states_rating,
            "check_sentiment": check.sentiment,
            "attempts": attempt + 1,
        }
        if result["consistent"]:
            break
    return result


def generate_reviews(items: pd.DataFrame, client: ChatClient, cache_path: Path, workers: int) -> pd.DataFrame:
    cache = GenerationCache(client, cache_path)
    records = items.to_dict("records")
    results: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(review_item, record, cache): index for index, record in enumerate(records)}
        for done, future in enumerate(as_completed(futures), start=1):
            results[futures[future]] = future.result()
            if done % 25 == 0 or done == len(futures):
                logger.info("Reviewed %d/%d, new LLM calls %d", done, len(futures), cache.api_calls)
    cache.flush()
    return pd.concat([items.reset_index(drop=True), pd.DataFrame([results[i] for i in range(len(records))])], axis=1)


def build_manifest(reviews: pd.DataFrame, args: argparse.Namespace, model: str) -> dict:
    consistent = reviews["consistent"]
    low = reviews[reviews["clarity"] == "low"]
    return {
        "prompt_version": prompts.PROMPT_VERSION,
        "sf": args.sf,
        "seed": args.seed,
        "n": len(reviews),
        "generator_model": model,
        "strata_mix": tpch_facts.STRATA_MIX,
        "late_days": tpch_facts.LATE_DAYS,
        "counts_truth_clarity": {
            f"{truth}/{clarity}": int(count)
            for (truth, clarity), count in reviews.groupby(["truth", "clarity"]).size().items()
        },
        "consistent_rate": float(consistent.mean()),
        "consistent_by_stratum": {k: float(v) for k, v in reviews.groupby("stratum")["consistent"].mean().items()},
        "failure_rates": {
            "unparseable": float(reviews["check_sentiment"].isna().mean()),
            "contradiction": float((reviews["check_contradiction"] == True).mean()),  # noqa: E712
            "states_rating": float((reviews["check_states_rating"] == True).mean()),  # noqa: E712
        },
        "sentiment_agreement_low": (
            float((low["check_sentiment"] == low["truth"]).mean()) if len(low) else None
        ),
        "mean_attempts": float(reviews["attempts"].mean()),
    }


def write_outputs(reviews: pd.DataFrame, manifest: dict, out: Path) -> None:
    reviews.to_parquet(out / REVIEWS, index=False)
    sample = [
        {"tuple_id": row.tuple_id, "summary": f"truth={row.truth} clarity={row.clarity} stratum={row.stratum}"}
        for row in reviews[reviews["consistent"]].itertuples()
    ]
    io.write_json(out / SAMPLE_IDS, {POPULATION: sample})
    io.write_json(out / MANIFEST, manifest)


def describe_example(item: dict) -> str:
    """The parameters behind one generation prompt, for the dry-run log."""
    facts = ", ".join(f"{name}={item[name]}" for name in (
        "returned", "days_late", "discount", "grade", "finish", "material", "container", "ship_mode",
    ))
    return "\n".join([
        f"Example item: {item['tuple_id']}  (order {item['l_orderkey']}, line {item['l_linenumber']})",
        f"  TPC-H facts:  {facts}",
        f"  design:       stratum={item['stratum']}, truth={item['truth']}, clarity={item['clarity']}",
        f"  generation:   prompt_version={prompts.PROMPT_VERSION}, "
        f"temperature={prompts.GENERATION_TEMPERATURE}, max_tokens={prompts.GENERATION_MAX_TOKENS}",
        f"  validation:   temperature={prompts.VALIDATION_TEMPERATURE}, "
        f"max_tokens={prompts.VALIDATION_MAX_TOKENS}, retries={MAX_RETRIES}",
    ])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="Working directory (database, cache, outputs).")
    parser.add_argument("--sf", type=float, default=0.01, help="TPC-H scale factor for --dbgen-only.")
    parser.add_argument("--n", type=int, default=300, help="Number of reviews.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dbgen-only", action="store_true", help="Build tpch.duckdb and stop (needs internet).")
    parser.add_argument("--dry-run", action="store_true", help="Sample and assign the design, no LLM calls.")
    parser.add_argument("--llm-model", default=None, help="Generator model, as served (e.g. llama-3.1-8b).")
    parser.add_argument("--llm-provider", choices=["anthropic", "openai"], default=None)
    parser.add_argument("--workers", type=int, default=8, help="Parallel LLM requests.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    return parser


def main(argv: list[str] | None = None, client: ChatClient | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    if args.dbgen_only:
        tpch_facts.build_database(out / DATABASE, args.sf)
        logger.info("Built %s (sf=%s)", out / DATABASE, args.sf)
        return

    facts = tpch_facts.load_facts(out / DATABASE)
    logger.info("Candidates per stratum: %s", facts["stratum"].value_counts().to_dict())
    items = tpch_facts.assign_design(tpch_facts.sample_items(facts, args.n, args.seed), args.seed)
    logger.info("Truth x clarity:\n%s", items.groupby(["truth", "clarity"]).size().unstack())

    if args.dry_run:
        items.to_parquet(out / DRY_RUN, index=False)
        example = items.iloc[0].to_dict()
        logger.info(
            "Dry run: wrote %s\n%s\nExample generation prompt:\n%s",
            out / DRY_RUN, describe_example(example), prompts.generation_prompt(example),
        )
        return

    if client is None:
        provider, model = config.resolve_llm(args.llm_model, args.llm_provider)
        client = make_client(provider, model)
    reviews = generate_reviews(items, client, out / CACHE, args.workers)
    manifest = build_manifest(reviews, args, client.model)
    write_outputs(reviews, manifest, out)
    logger.info("Wrote %s; consistent %.1f%%", out / REVIEWS, 100 * manifest["consistent_rate"])


if __name__ == "__main__":
    main()

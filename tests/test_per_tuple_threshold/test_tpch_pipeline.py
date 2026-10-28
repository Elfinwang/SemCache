import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import per_tuple_threshold
from per_tuple_threshold.pipeline import io
from per_tuple_threshold.pipeline.predict import PredictionConfig
from per_tuple_threshold.prediction.model import TrainingConfig
from per_tuple_threshold.schema import ThresholdEstimate, TupleCalibration
from per_tuple_threshold.tpch import analysis, loader
from per_tuple_threshold.tpch import pipeline as tpch_pipeline
from per_tuple_threshold.tpch.features import FEATURE_NAMES, TpchFeatureExtractor
from per_tuple_threshold.tpch.predicates import PREDICATES
from per_tuple_threshold.tpch.query_generation import generation_prompt

# Each review's answer flips for q' below this similarity, so t(d) should land near it.
CUTOFFS = {"high": 0.45, "medium": 0.65, "low": 0.85}
DIM = 64


def _unit(seed_text: str) -> np.ndarray:
    seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:8], 16)
    vector = np.random.default_rng(seed).normal(size=DIM)
    return vector / np.linalg.norm(vector)


class FakeEmbedder:
    """Queries tagged ``(sim=x)`` sit at cosine x from the base query; everything else is random."""

    def __init__(self):
        self.base = np.eye(DIM)[0]

    def encode_queries(self, texts):
        out = {}
        for text in texts:
            match = re.search(r"sim=([0-9.]+)", text)
            if match is None:
                out[text] = self.base
                continue
            sim = float(match.group(1))
            other = _unit(text)
            other = other - other @ self.base * self.base
            other /= np.linalg.norm(other)
            out[text] = sim * self.base + np.sqrt(1 - sim**2) * other
        return out

    def encode_documents(self, texts):
        return {text: _unit(text) for text in texts}


def _pool_json() -> str:
    def rows(name, sims):
        return [f"Is this a favourable product review, {name} {i} (sim={sim:.2f})?" for i, sim in enumerate(sims)]

    return json.dumps({
        "paraphrases": rows("paraphrase", np.linspace(0.95, 0.60, 16)),
        "intensity": rows("intensity", np.linspace(0.80, 0.50, 8)),
        "should_reuse": rows("reuse", np.linspace(0.86, 0.70, 6)),
        "should_not_reuse": rows("noreuse", np.linspace(0.60, 0.30, 8)),
        "unrelated": rows("unrelated", np.linspace(0.40, 0.10, 8)),
    })


class FakeClient:
    model = "fake"
    provider = "local"

    def complete(self, messages, max_tokens, temperature):
        prompt = messages[0]["content"]
        if prompt.startswith("Generate calibration queries"):
            return _pool_json()
        sim_match = re.search(r"sim=([0-9.]+)", prompt)
        sim = float(sim_match.group(1)) if sim_match else 1.0
        cutoff = float(re.search(r"cutoff=([0-9.]+)", prompt).group(1))
        cached = "YES" if "truth=positive" in prompt else "NO"
        flipped = "NO" if cached == "YES" else "YES"
        return cached if sim >= cutoff else flipped


def _write_tpch_dir(path: Path) -> Path:
    rows, sample = [], []
    for clarity, cutoff in CUTOFFS.items():
        for i in range(4):
            truth = "positive" if i % 2 == 0 else "negative"
            tuple_id = f"tpch:{clarity}:{i}"
            jitter = 0.02 * (i - 1.5)
            text = f"Review {i} truth={truth} cutoff={cutoff + jitter:.3f} about a {clarity} part."
            rows.append({"tuple_id": tuple_id, "text": text, "truth": truth, "clarity": clarity,
                         "stratum": "on_time" if truth == "positive" else "late", "consistent": True})
            sample.append({"tuple_id": tuple_id, "summary": f"truth={truth} clarity={clarity}"})
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path / loader.REVIEWS, index=False)
    (path / loader.SAMPLE_IDS).write_text(json.dumps({"tpch_reviews": sample}))
    (path / "manifest.json").write_text(json.dumps({"prompt_version": "gen-v1", "generator_model": "fake-gen"}))
    return path


def test_loader_builds_tuples_in_sample_order(tmp_path):
    populations = loader.load_tpch_populations(_write_tpch_dir(tmp_path / "data"))
    tuples = populations["tpch_reviews"]
    assert [item.tuple_id for item in tuples][:2] == ["tpch:high:0", "tpch:high:1"]
    assert tuples[0].context.startswith("Review: Review 0")
    assert loader.load_design(tmp_path / "data")["tpch:low:3"]["clarity"] == "low"


def test_loader_rejects_unknown_sample_ids(tmp_path):
    data = _write_tpch_dir(tmp_path / "data")
    (data / loader.SAMPLE_IDS).write_text(json.dumps({"tpch_reviews": [{"tuple_id": "tpch:x:1", "summary": ""}]}))
    with pytest.raises(KeyError, match="missing"):
        loader.load_tpch_populations(data)


def test_generation_prompt_is_about_product_reviews():
    prompt = generation_prompt(PREDICATES[0])
    assert "industrial part" in prompt and "movie" not in prompt.lower()
    for key in ("paraphrases", "intensity", "should_reuse", "should_not_reuse", "unrelated"):
        assert f'"{key}"' in prompt


def test_feature_extractor_shape_and_names():
    extractor = TpchFeatureExtractor(FakeEmbedder())
    features = extractor.transform(["Great part, arrived early!", "Late and returned."], ["YES", "NO"], "yes_no")
    assert features.shape == (2, len(FEATURE_NAMES))
    assert features[0, 1] > 0 and features[1, 3] > 0  # exclamations, negative lexicon
    assert features[0, -1] == 1.0 and features[1, -1] == 0.0


@pytest.mark.parametrize(("reason", "expected"), [
    ("explore: single label class", "all_match"),
    ("explore: logistic regression failed", "fit_failed"),
    ("explore: fewer than 6 observations", "too_few"),
])
def test_bucket_follows_estimator_reason(reason, expected):
    record = TupleCalibration("t", "", "", "YES", ThresholdEstimate(None, None, None, reason), ())
    assert analysis.bucket(record) == expected
    assert analysis.effective_threshold(record) == (0.0 if expected == "all_match" else None)


def test_end_to_end_recovers_clarity_ordering(tmp_path, monkeypatch):
    monkeypatch.setattr(tpch_pipeline, "make_client", lambda provider, model: FakeClient())
    monkeypatch.setattr("per_tuple_threshold.llm.labeler.make_client", lambda provider, model: FakeClient())
    data = _write_tpch_dir(tmp_path / "data")
    run_dir = tmp_path / "run"

    tpch_pipeline.run_tpch_calibration(
        tpch_pipeline.TpchCalibrationConfig(
            output_dir=run_dir, tpch_dir=data, embedding_model="fake",
            llm_provider="local", llm_model="fake", workers=2,
        ),
        embedder=FakeEmbedder(),
    )
    for name in (io.RUN_CONFIG, io.POOLS, io.LABEL_CACHE):
        assert (run_dir / name).is_file()
    assert io.read_json(run_dir / io.RUN_CONFIG)["dataset"] == "tpch"

    clarity = io.read_json(run_dir / io.CALIBRATION_DIR / tpch_pipeline.CLARITY)["tpch_positive"]["by_clarity"]
    means = {name: clarity[name]["mean_t_fitted"] for name in ("high", "medium", "low")}
    assert all(value is not None for value in means.values())
    assert means["high"] < means["medium"] < means["low"]

    tpch_pipeline.run_tpch_prediction(
        PredictionConfig(run_dir=run_dir, embedding_model="fake", training=TrainingConfig(epochs=30)),
        embedder=FakeEmbedder(),
    )
    result = io.read_json(run_dir / io.PREDICTION_DIR / "tpch_positive.json")
    assert result["status"] == "ok" and result["predicted_by_clarity"]
    metrics = io.read_json(run_dir / io.PREDICTION_DIR / io.METRICS)
    assert metrics["feature_names"] == list(FEATURE_NAMES) and metrics["dataset"] == "tpch"


def test_core_does_not_import_tpch():
    root = Path(per_tuple_threshold.__file__).parent
    for path in root.rglob("*.py"):
        if "tpch" in path.relative_to(root).parts[:2] or path.parent.name == "tpch_reviews":
            continue
        assert "per_tuple_threshold.tpch" not in path.read_text(), path

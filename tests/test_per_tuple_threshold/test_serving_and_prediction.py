import numpy as np
import pytest

from per_tuple_threshold.evaluation import score_decisions
from per_tuple_threshold.llm.labeler import parse_label
from per_tuple_threshold.pipeline.predict import split_tuples
from per_tuple_threshold.prediction.margin import MAX_MARGIN, safety_margin
from per_tuple_threshold.prediction.model import ThresholdPredictor, TrainingConfig
from per_tuple_threshold.schema import KIND_SCORE, KIND_YES_NO, Observation, ThresholdEstimate, TupleCalibration
from per_tuple_threshold.serving import PerTupleSemanticFilter, safe_threshold, should_reuse


def test_should_reuse():
    assert should_reuse(0.8, 0.8)
    assert not should_reuse(0.79, 0.8)
    assert not should_reuse(1.0, None)


def test_safe_threshold_clips():
    assert safe_threshold(0.95, 0.1) == 1.0
    assert safe_threshold(0.5, 0.1) == pytest.approx(0.6)
    assert safe_threshold(None, 0.1) is None


def test_filter_partitions_tuples():
    decision = PerTupleSemanticFilter({"a": 0.3, "b": 0.7, "c": None}, margin=0.1).decide(0.5)
    assert decision.reuse == ["a"]
    assert decision.recompute == ["b", "c"]


def test_safety_margin_quantile_and_cap():
    truth = np.array([0.5, 0.5, 0.5, 0.5])
    assert safety_margin(truth, truth) == 0.0
    assert safety_margin(truth, truth - 0.05, delta=0.1) == pytest.approx(0.05)
    assert safety_margin(truth, truth - 0.9) == MAX_MARGIN


@pytest.mark.parametrize(
    ("text", "kind", "expected"),
    [("YES", KIND_YES_NO, "YES"), ("no.", KIND_YES_NO, "NO"), ("4", KIND_SCORE, "4"), ("Score: 3", KIND_SCORE, "3")],
)
def test_parse_label(text, kind, expected):
    assert parse_label(text, kind) == expected


def test_parse_label_rejects_unparseable():
    with pytest.raises(ValueError):
        parse_label("maybe", KIND_YES_NO)


def _observation(similarity, match):
    return Observation("n", "eval", "paraphrase", "q", similarity, None, "YES", match)


def test_score_decisions():
    score = score_decisions([
        (_observation(0.9, True), 0.5),
        (_observation(0.9, False), 0.5),
        (_observation(0.4, False), 0.5),
        (_observation(0.4, True), 0.5),
    ])
    assert score["reuses"] == 2
    assert score["false_reuses"] == 1
    assert score["correct_rejects"] == 1
    assert score["precision"] == 0.5
    assert score["decision_accuracy"] == 0.5


def _record(tuple_id):
    return TupleCalibration(tuple_id, "", "", "YES", ThresholdEstimate(0.5, 1.0, None, "ok"), ())


def test_split_keeps_duplicates_together_and_is_seeded():
    records = [_record(f"t{i}") for i in range(9)] + [_record("t0")]
    calibration, target = split_tuples(records, 2 / 3, seed=0)
    calibration_ids = {r.tuple_id for r in calibration}
    target_ids = {r.tuple_id for r in target}
    assert not calibration_ids & target_ids
    assert len(calibration_ids) == 6
    assert split_tuples(records, 2 / 3, seed=0)[0] == calibration


def test_predictor_learns_a_linear_threshold():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(40, 3)).astype(np.float32)
    thresholds = (0.5 + 0.15 * np.tanh(features[:, 0])).astype(np.float32)
    predictor = ThresholdPredictor(3, TrainingConfig(epochs=300))
    history = predictor.fit(features, thresholds)
    assert history[-1] < history[0]
    predicted = predictor.predict(features)
    assert np.all((predicted >= 0) & (predicted <= 1))
    assert np.mean(np.abs(predicted - thresholds)) < 0.05

import numpy as np
import pytest

from per_tuple_threshold.calibration.calibrator import calibrate_predicate, pooled_threshold
from per_tuple_threshold.calibration.query_generation import clean_pool, is_adjective_swap, parse_json_block
from per_tuple_threshold.calibration.threshold import estimate_threshold
from per_tuple_threshold.calibration.variant_selection import select_variants
from per_tuple_threshold.schema import KIND_YES_NO, ROLE_EVAL, ROLE_FIT, DataTuple, Predicate, QueryVariant
from vcache.vcache_policy.strategies.verified import _Algorithm

BASE = "Determine if the following movie review is clearly positive."


def test_threshold_matches_vcache_estimator():
    sims = np.array([1.0, 0.93, 0.85, 0.77, 0.6, 0.5, 0.4, 0.3])
    labels = np.array([1, 1, 1, 0, 1, 0, 0, 0])
    expected_t, expected_gamma, _ = _Algorithm(delta=0.01)._estimate_parameters(sims, labels)
    estimate = estimate_threshold(sims, labels)
    assert estimate.reason == "ok"
    assert estimate.t_hat == expected_t
    assert estimate.gamma == expected_gamma
    assert 0.0 <= estimate.t_hat <= 1.0


def test_threshold_explores_with_too_few_observations():
    assert estimate_threshold([1.0, 0.9, 0.5], [1, 1, 0]).t_hat is None


def test_threshold_explores_with_single_label_class():
    estimate = estimate_threshold([1.0, 0.9, 0.8, 0.7, 0.6, 0.5], [1] * 6)
    assert estimate.t_hat is None
    assert "single label class" in estimate.reason


@pytest.mark.parametrize(
    "candidate",
    [
        "Determine if the following movie review is clearly negative.",
        "Determine if the following movie review is not clearly positive.",
    ],
)
def test_adjective_swap_is_rejected(candidate):
    assert is_adjective_swap(BASE, candidate)


def test_paraphrase_is_not_an_adjective_swap():
    assert not is_adjective_swap(BASE, "Is this review a clearly positive assessment of the movie?")


def test_clean_pool_drops_base_duplicates_and_swaps():
    parsed = parse_json_block(
        '```json\n{"paraphrases": ["%s", "Is the review upbeat?", "Is the review upbeat?"],'
        ' "unrelated": ["Who directed it?"],'
        ' "should_not_reuse": ["Determine if the following movie review is clearly negative."]}\n```'
        % BASE
    )
    pool, rejected = clean_pool(BASE, parsed)
    assert pool["paraphrase"] == ["Is the review upbeat?"]
    assert pool["unrelated"] == ["Who directed it?"]
    assert pool["should_not_reuse"] == []
    assert rejected == ["Determine if the following movie review is clearly negative."]


def test_select_variants_hits_targets_and_roles():
    rng = np.random.default_rng(0)
    base_vector = np.array([1.0, 0.0])

    def at(similarity):
        return np.array([similarity, np.sqrt(1 - similarity**2)])

    pool = {
        "paraphrase": [f"p{i}" for i in range(14)],
        "intensity": ["i0", "i1", "i2"],
        "should_reuse": ["r0", "r1"],
        "should_not_reuse": ["n0", "n1", "n2"],
        "unrelated": [f"u{i}" for i in range(6)],
    }
    vectors = {BASE: base_vector}
    for texts in pool.values():
        for text in texts:
            vectors[text] = at(float(rng.uniform(0.2, 0.95)))
    variants = select_variants(pool, vectors, BASE)
    assert sum(item.role == ROLE_FIT for item in variants) == 20
    assert sum(item.role == ROLE_EVAL for item in variants) == 6
    assert len({item.text for item in variants}) == len(variants)
    unrelated_fit = [item.similarity for item in variants if item.category == "unrelated" and item.role == ROLE_FIT]
    unrelated_eval = [item.similarity for item in variants if item.category == "unrelated" and item.role == ROLE_EVAL]
    assert max(unrelated_fit) <= min(unrelated_eval)


def test_calibrate_predicate_uses_fit_observations_only():
    predicate = Predicate(id="p", kind=KIND_YES_NO, population="general", base=BASE)
    sims = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    variants = [
        QueryVariant(name=f"v{i}", role=ROLE_FIT, category="paraphrase", text=f"q{i}", similarity=s)
        for i, s in enumerate(sims)
    ] + [QueryVariant(name="e", role=ROLE_EVAL, category="paraphrase", text="qe", similarity=0.95)]
    item = DataTuple(tuple_id="m:1", summary="POSITIVE", context="Review: great", text="great")
    answers = {("yes_no", "m:1", BASE): "YES", ("yes_no", "m:1", "qe"): "NO"}
    for i, s in enumerate(sims):
        answers[("yes_no", "m:1", f"q{i}")] = "YES" if s >= 0.7 else "NO"
    [record] = calibrate_predicate(predicate, [item], variants, answers)
    assert record.cached_answer == "YES"
    assert len(record.fit_observations()) == 7
    assert [obs.match for obs in record.eval_observations()] == [False]
    assert record.threshold.t_hat is not None
    assert 0.6 <= record.threshold.t_hat <= 0.7
    assert pooled_threshold([record]).t_hat == record.threshold.t_hat

import json

import pandas as pd
import pytest

from per_tuple_threshold.data.tpch_reviews import facts as F
from per_tuple_threshold.data.tpch_reviews import generate as G
from per_tuple_threshold.data.tpch_reviews import prompts as P


def _facts(per_stratum: int = 40) -> pd.DataFrame:
    rows = []
    for stratum, returned, days_late in (("on_time", False, -3), ("returned", True, 2), ("late", False, 12)):
        for i in range(per_stratum):
            rows.append({
                "tuple_id": f"tpch:{stratum}:{i}", "l_orderkey": i, "l_linenumber": 1,
                "p_partkey": i, "s_suppkey": i, "c_custkey": i,
                "returned": returned, "days_late": days_late, "discount": 0.09 if i % 2 else 0.02,
                "unit_price": 100.0, "grade": "economy", "finish": "anodized", "material": "steel",
                "container": "LG BAG", "ship_mode": "AIR", "stratum": stratum,
            })
    return pd.DataFrame(rows)


class FakeClient:
    """Canned quoted reviews; validation always agrees with the facts."""

    model = "fake-model"
    provider = "openai"

    def __init__(self):
        self.calls = 0

    def complete(self, messages, max_tokens, temperature):
        self.calls += 1
        prompt = messages[0]["content"]
        if prompt.startswith("You are a business customer"):
            return f'"Review {self.calls} about the part."'
        sentiment = "POSITIVE" if "arrived on or ahead" in prompt else "NEGATIVE"
        return f"CONTRADICTS: NO\nRATING: NO\nSENTIMENT: {sentiment}"


def test_stratum_counts_follow_mix():
    assert F.stratum_counts(300) == {"on_time": 150, "returned": 75, "late": 75}
    assert sum(F.stratum_counts(30).values()) == 30


def test_sample_and_design_are_balanced_and_seeded():
    items = F.assign_design(F.sample_items(_facts(), 60, seed=0), seed=0)
    assert items["stratum"].value_counts().to_dict() == {"on_time": 30, "returned": 15, "late": 15}
    assert set(items.loc[items.stratum == "on_time", "truth"]) == {"positive"}
    assert set(items.loc[items.stratum != "on_time", "truth"]) == {"negative"}
    cells = items.groupby(["truth", "clarity"]).size()
    assert cells.min() == cells.max() == 10
    again = F.assign_design(F.sample_items(_facts(), 60, seed=0), seed=0)
    assert items["tuple_id"].tolist() == again["tuple_id"].tolist()
    assert items["clarity"].tolist() == again["clarity"].tolist()


def test_sample_rejects_too_small_stratum():
    with pytest.raises(ValueError, match="raise --sf"):
        F.sample_items(_facts(per_stratum=5), 60, seed=0)


def test_load_facts_points_at_dbgen_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="--dbgen-only"):
        F.load_facts(tmp_path / "missing.duckdb")


def test_prompts_describe_facts_without_codes():
    item = {**_facts().iloc[45].to_dict(), "truth": "negative", "clarity": "low"}
    prompt = P.generation_prompt(item)
    assert "an economy-grade anodized steel part" in prompt
    assert "a large bag" in prompt
    assert "returned it" in prompt and "dissatisfied" in prompt
    assert "LG BAG" not in prompt and "tpch:" not in prompt


def test_parse_validation_and_pass_rule():
    result = P.parse_validation("CONTRADICTS: NO\nRATING: NO\nSENTIMENT: POSITIVE")
    assert result.passes("positive", "high")
    assert not result.passes("negative", "medium")
    assert result.passes("negative", "low")
    with pytest.raises(ValueError):
        P.parse_validation("I think it is fine.")


def test_failed_validation_retries_with_new_cache_entry(tmp_path):
    class Flaky(FakeClient):
        def complete(self, messages, max_tokens, temperature):
            self.calls += 1
            if messages[0]["content"].startswith("You are a business customer"):
                return "A review."
            return "RATING: YES" if self.calls <= 2 else "CONTRADICTS: NO\nRATING: NO\nSENTIMENT: NEGATIVE"

    item = {**_facts().iloc[45].to_dict(), "truth": "negative", "clarity": "high"}
    client = Flaky()
    result = G.review_item(item, G.GenerationCache(client, tmp_path / "cache.json"))
    assert result["consistent"] and result["attempts"] == 2
    assert client.calls == 4


def test_generate_end_to_end_with_fake_client(tmp_path):
    items = F.assign_design(F.sample_items(_facts(), 30, seed=1), seed=1)
    client = FakeClient()
    reviews = G.generate_reviews(items, client, tmp_path / G.CACHE, workers=4)
    args = G.build_parser().parse_args(["--out", str(tmp_path), "--n", "30"])
    manifest = G.build_manifest(reviews, args, client.model)
    G.write_outputs(reviews, manifest, tmp_path)

    saved = pd.read_parquet(tmp_path / G.REVIEWS)
    assert len(saved) == 30 and saved["consistent"].all()
    assert not saved["text"].str.startswith('"').any()
    sample = json.loads((tmp_path / G.SAMPLE_IDS).read_text())[G.POPULATION]
    assert len(sample) == 30 and sample[0]["summary"].startswith("truth=")
    assert manifest["consistent_rate"] == 1.0 and manifest["prompt_version"] == P.PROMPT_VERSION

    calls = client.calls
    G.generate_reviews(items, client, tmp_path / G.CACHE, workers=4)
    assert client.calls == calls  # resumed entirely from cache

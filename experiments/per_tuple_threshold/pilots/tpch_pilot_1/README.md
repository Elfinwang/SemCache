# TPC-H reviews: pilot 1 (30 reviews)

A small, self-contained run of the per-tuple threshold pipeline on generated TPC-H
product reviews. Everything needed to re-run calibration and prediction is here,
including every LLM answer, so re-running needs **no API key and no GPU**.

**This is a pilot (n = 30). It shows the pipeline end to end; the numbers are not
conclusive.**

## What's in it

```
data/                       the 30 reviews (input)
  reviews.parquet           text + design columns: truth, clarity, stratum
  tpch_sample_ids.json      the population "tpch_reviews" (all 30)
  manifest.json             how the reviews were generated
run/                        output of `python -m per_tuple_threshold.tpch calibrate / predict`
  generated_pools.json      the 46 reworded questions q' for tpch_positive
  llm_cache.json            all 810 LLM answers (30 reviews x 27 questions)
  calibration/              t(d) per review, held-out scores, clarity.json (by-clarity breakdown)
  prediction/               predicted t(d), margin, held-out reuse scores
SUMMARY.txt                 the three result tables
tools/                      exploration scripts (no LLM calls)
```

## Setup

| Step | Setting |
|---|---|
| Reviews | Llama-3.1-8B-Instruct (vLLM, Compute Canada Fir), prompt `gen-v1` |
| Labels (LLM answers) | `qwen/qwen3.8-27b` via Groq, temperature 0 |
| Similarity s(q, q') | `sentence-transformers/all-MiniLM-L6-v2` |
| Predicate | `tpch_positive`: "Determine if the following product review is clearly positive." |
| Design | `truth` from TPC-H facts (returned or >= 7 days late -> negative, on time -> positive); `clarity` (high / medium / low) set when generating |

## Results (see SUMMARY.txt)

- **Negative reviews (14):** every reworded question got the same answer (NO), so no threshold can be fitted; reuse is always safe for them.
- **Positive reviews (16):** mean t(d) is high 0.60, medium 0.60, low 0.55. There's no rise for low clarity at this n.
- **Held-out reuse on 10 reviews (60 questions):**
  - one global threshold: 60/60 reused, 24 wrong;
  - per-review predicted thresholds: 20–22/60 reused, 0 wrong.
- **Per review, thresholds follow how hedged the text actually is,** not the requested `clarity`. Compare `tpch:pilot:20` (0.54) with `tpch:pilot:21` (0.79), both labelled "low".
- **`tpch:pilot:23` is a failure case:** truthfully positive, the LLM says not *clearly* positive, and its t(d) comes out 0.00 although "does it express approval?" flips to YES. Similarity can't tell a weaker question from a stronger one.

## Re-run (from the repo root, with the repo venv)

```bash
B=experiments/per_tuple_threshold/pilots/tpch_pilot_1
# calibrate: reuses the saved questions and answers -> 0 LLM calls, identical results
python -m per_tuple_threshold.tpch calibrate --tpch-dir $B/data \
    --llm-provider local --llm-model qwen/qwen3.8-27b \
    --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
    --pools $B/run/generated_pools.json --label-cache $B/run/llm_cache.json --output /tmp/tpch_rerun
python -m per_tuple_threshold.tpch predict --run-dir /tmp/tpch_rerun                       # add --all-match-as-zero to keep all-match reviews
```
`--llm-model` must stay `qwen/qwen3.8-27b`: the saved answers are keyed by model name.
Changing the questions, model or reviews makes new LLM calls (set `LLM_BASE_URL` and
`LLM_API_KEY`, see `per_tuple_threshold/README.md`).

## Explore

```bash
T=experiments/per_tuple_threshold/pilots/tpch_pilot_1/tools
python $T/overview.py --truth positive --clarity low --text   # one line per review (+ text): flips per question type
python $T/review.py tpch:pilot:23 --full                      # one review, every question, sorted by similarity
python $T/refit.py --drop unrelated should_not_reuse          # re-fit t(d) without some question types
```

## Caveats

- **Synthetic reviews (`gen-v1`) are formulaic,** and some "low clarity" reviews still state their feeling. `clarity` is what the generator was asked for, not a measured property.
- **The tuple ids `tpch:pilot:NN` are stand-ins.** The dataset was rebuilt from the printed generator output, which didn't include the TPC-H order and line keys.
- **"Match" means the LLM agrees with itself,** not that it is correct. `truth` is not yet used to score the LLM's answers.
- **Small embedding model, 30 reviews.**

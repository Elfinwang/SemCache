# Per-Tuple Similarity Thresholds for Semantic Caching

This package extends [vCache](../README.md) with **per-tuple** reuse thresholds.
A cached answer `a_d = LLM(q, d)` is reused for a new query `q'` when

```
s(q, q') >= t(d)
```

where `t(d)` is specific to the data tuple `d` rather than one global value.
Thresholds are calibrated on a small subset of tuples with LLM-generated query
variants and **vCache's original threshold estimator**, then predicted for all
other tuples by a small neural network `g_theta`.

The experiments run on the **movie** scenario of [SemBench](https://github.com/SemBench/SemBench).

---

## Contents

1. [Method](#1-method)
2. [How the threshold predictor is trained](#2-how-the-threshold-predictor-is-trained)
3. [Repository layout](#3-repository-layout)
4. [Environment setup](#4-environment-setup)
5. [External resources: SemBench, LLM API key, embedding model](#5-external-resources)
6. [Running the pipeline](#6-running-the-pipeline)
7. [Output files](#7-output-files)
8. [Configuration reference](#8-configuration-reference)
9. [Implementation notes and limitations](#9-implementation-notes-and-limitations)
10. [Tests](#10-tests)

---

## 1. Method

### 1.1 Motivation: why per-tuple similarity thresholds?

- vCache reuses cached LLM answers for semantically similar queries.
- The semantic filter decides whether a new query `q'` is similar enough to the
  original query `q` to reuse the cached answer: `s(q, q') >= t`, where
  `s(q, q')` is semantic similarity and `t` a threshold.
- A single global threshold is suboptimal because different tuples have
  different sensitivity to query wording.

Example:

| | |
|---|---|
| Original query `q` | "Find positive reviews about the product." |
| New query `q'` | "Find enthusiastic reviews about the product." |
| Tuple `d1` | "It's okay, works fine." (positive, but not enthusiastic) |
| Tuple `d2` | "Absolutely love it! Best purchase ever!" (positive and enthusiastic) |

For `d1` the answer changes between `q` and `q'`, so reuse must be blocked unless
the two queries are almost identical. For `d2` the answer does not change, so
reuse is safe even at a lower similarity. Hence `t(d1) > t(d2)`, and the filter
needs a per-tuple threshold `t(d)`.

### 1.2 Problem setting

- A fixed base query `q` and a dataset `D = {d_1, ..., d_M}`.
- For every tuple the answer `a_d = LLM(q, d)` is computed and cached.
- A new query `q'` arrives: a paraphrase, an opposite query, a stronger or
  weaker version of the intent, or an unrelated question.
- Decision: reuse `a_d` if `s(q, q') >= t(d)`, otherwise call `LLM(q', d)`.
- Core problem: derive `t(d)` for all `M` tuples without calling the LLM too often.

### 1.3 The scaling challenge

Calibrating one tuple vCache-style needs `K` query variants and `K` LLM calls.
Doing this for every tuple costs `O(M * K)`, which is infeasible for millions of
tuples. The approach amortises: spend LLM calls on a small calibration set and
predict thresholds for the rest.

### 1.4 Approach: learn a network that predicts the threshold

1. Select a calibration set `C ⊂ D`, `|C| = N << M`.
2. For each `d_i ∈ C`, estimate its threshold `t(d_i)` from query variants (1.5).
3. Extract observable features `phi(d)` that cost no extra LLM call
   (`prediction/features.py`).
4. Train `g_theta : phi(d) -> t(d)` on `{(phi(d_i), t(d_i)) : d_i ∈ C}`
   (`prediction/model.py`).
5. Predict `t_hat(d) = g_theta(phi(d))` for every remaining tuple.
6. Add a conservative margin `m`: `t_safe(d) = clip(t_hat(d) + m, 0, 1)`
   (`prediction/margin.py`).
7. Serve: reuse `a_d` iff `s(q, q') >= t_safe(d)` (`serving.py`).

### 1.5 Constructing query variants `q'`

Implemented in `calibration/query_generation.py` and `calibration/variant_selection.py`.

**Generation.** For each base query the LLM (Claude Sonnet 4.5, temperature 0.7)
writes one JSON pool with five categories:

| Category | Count requested | Meaning |
|---|---:|---|
| `paraphrase` | 16 | same meaning, different wording |
| `intensity` | 8 | same attribute, weaker or stronger degree ("positive" vs. "enthusiastic") |
| `should_reuse` | 6 | the cached answer to `q` should be reused |
| `should_not_reuse` | 8 | different information need; the cached answer should not be reused |
| `unrelated` | 8 | a different fact about the same review(s) |

Candidates that only flip one polarity word of `q` (e.g. "clearly positive" ->
"clearly negative") are rejected by `is_adjective_swap`, and the generator is
re-prompted with the rejected strings as negative examples (at most 3 attempts).

**Selection at controlled similarity levels.** `q` and every candidate are
embedded; candidates are picked closest to fixed cosine targets (the generator
never sees the targets):

| Role | Category | Target `s(q, q')` |
|---|---|---|
| fit | paraphrase | 0.93, 0.89, 0.85, 0.81, 0.77, 0.73, 0.69, 0.65 |
| fit | intensity | 0.78, 0.66, 0.54 |
| fit | should_reuse | 0.84, 0.72 |
| fit | should_not_reuse | 0.58, 0.44, 0.32 |
| fit | unrelated | the 4 lowest-similarity candidates |
| eval | paraphrase | 0.91, 0.83, 0.75, 0.67 |
| eval | unrelated | the next 2 lowest-similarity candidates |

`fit` variants (plus `q` itself at `s = 1`) are used to estimate `t(d)`, i.e.
`K = 20` LLM calls per calibration tuple. `eval` variants are held out and only
used to score reuse decisions.

**Labels.** For each tuple and variant, `LLM(q', d)` is requested at temperature
0 (`llm/labeler.py`). An observation is `(s(q, q'), 1[LLM(q', d) == a_d])`.

**Threshold.** `calibration/threshold.py` calls vCache's own
`_Algorithm._estimate_parameters` (`vcache/vcache_policy/strategies/verified.py`),
unchanged:

```
P(match | s) = sigmoid(a + gamma * s)     # unpenalised logistic regression
gamma        = max(gamma, 1e-6)
t(d)         = clip(-a / gamma, 0, 1)      # rounded to 3 decimals
```

As in `_Algorithm.select_action`, fewer than 6 observations or a failed fit
(including a single label class) yields no threshold, and that tuple is never reused.

### 1.6 Serving with a per-tuple semantic filter

For a new query `q'`, compute `s(q, q')` once, then for each tuple compare it with
`t_safe(d)`: reuse `a_d` if it passes, otherwise call `LLM(q', d)`.
`PerTupleSemanticFilter.decide(similarity)` returns the two tuple lists.

### 1.7 Cost

- Naive per-tuple calibration: `M * (K + 1)` calls.
- This approach: `M` base answers plus `N * K` calibration labels, i.e. `M + N * K`,
  which is essentially `M` when `N << M`.

The `predict` stage reports both numbers per predicate (`cost` in its output).

---

## 2. How the threshold predictor is trained

This section describes exactly what `python -m per_tuple_threshold predict` does
(`pipeline/predict.py`). It runs **once per base query** (predicate); every
predicate gets its own `g_theta`.

1. **Input.** The stage-1 results: for every sampled tuple `d`, its cached answer
   `a_d`, its fitted `t(d)` (or `None`), and all labeled observations.
2. **Split.** Unique tuple ids are shuffled with `--seed` (default 0). The first
   `round(fraction * n)` ids form the calibration set `C` (`--calibration-fraction`,
   default 2/3); the rest are *target* tuples that play the role of the `M - N`
   uncalibrated tuples. With the frozen sample this is 16/24 tuples for the
   `general` population, 8/12 for `taken_3` and `ant_man_pairs`, and 9 of the 14
   unique tuples for `ant_man`.
3. **Training set.** Tuples in `C` that have a fitted `t(d)`; tuples whose fit
   returned `None` are excluded. A predicate with fewer than 3 such tuples is skipped.
4. **Features `phi(d)`** (8 values, no LLM call):
   `log_word_count`, `exclamations`, `lexicon_positive`, `lexicon_negative`,
   cosine of the tuple embedding to three anchor reviews (`sim_enthusiastic`,
   `sim_mild`, `sim_negative`), and `cached_answer` (1/0 for YES/NO,
   `(score - 1) / 4` for 1-5 scores). The tuple text is embedded with the same
   embedding model as the queries, without the query prompt.
5. **Model.** `ThresholdNet`: `Linear(8, 16) -> ReLU -> Linear(16, 1) -> sigmoid`.
   Features are standardised with the training mean and standard deviation.
   Full-batch Adam, learning rate 0.01, weight decay 0.01, 400 epochs, MSE loss
   against `t(d)`.
6. **Margin.** Leave-one-out on the training set: for each training tuple a fresh
   model is trained on the others and predicts it. The margin is the 90 % quantile
   (`--margin-delta 0.1`) of `t(d) - t_hat_loo(d)`, clipped to `[0, 0.2]`.
7. **Prediction.** The model trained on the full training set predicts `t_hat(d)`
   for every target tuple; `t_safe(d) = clip(t_hat(d) + m, 0, 1)`.
8. **Evaluation.** On the held-out (`eval`) variants of the target tuples
   (4 paraphrases and 2 unrelated queries each) four rules are scored:
   `predicted` (`t_safe`), `predicted_no_margin` (`t_hat`), `oracle` (the target
   tuple's own `t(d)` from stage 1), and `global` (one `t(q)` pooled over `C`).
   The target tuples' labels are used only for this evaluation; a deployment
   would not pay for them.

---

## 3. Repository layout

```
vCache/
├── vcache/                          # upstream vCache (provides the threshold estimator)
├── per_tuple_threshold/             # this package
│   ├── __main__.py                  # `python -m per_tuple_threshold`
│   ├── cli.py                       # argument parsing for the two stages
│   ├── config.py                    # default paths, model ids, environment variable names
│   ├── schema.py                    # dataclasses: Predicate, DataTuple, QueryVariant, ...
│   ├── predicates.py                # the 17 base queries q (SemBench movie + extras)
│   ├── data/
│   │   ├── sembench.py              # Reviews.csv loader, tuple construction
│   │   └── sample_ids.json          # frozen tuple sample per population
│   ├── llm/
│   │   ├── client.py                # Anthropic / OpenAI chat clients (keys from the environment)
│   │   └── labeler.py               # cached, parallel LLM(q', d) labeling
│   ├── embedding.py                 # sentence embeddings, s(q, q') = cosine
│   ├── calibration/
│   │   ├── query_generation.py      # LLM-generated q' pools
│   │   ├── variant_selection.py     # pick q' at controlled similarity levels
│   │   ├── threshold.py             # wrapper around vCache's _estimate_parameters
│   │   └── calibrator.py            # per-tuple t(d) and pooled t(q)
│   ├── prediction/
│   │   ├── features.py              # phi(d)
│   │   ├── model.py                 # ThresholdNet, ThresholdPredictor
│   │   └── margin.py                # safety margin
│   ├── serving.py                   # reuse rule, PerTupleSemanticFilter
│   ├── evaluation.py                # held-out scoring of reuse rules
│   ├── pipeline/
│   │   ├── calibrate.py             # stage 1
│   │   ├── predict.py               # stage 2
│   │   └── io.py                    # run-directory layout, JSON I/O
│   └── requirements.txt
├── tests/test_per_tuple_threshold/       # unit tests (no GPU, no network)
└── experiments/per_tuple_threshold/
    └── results/                     # run outputs (git-ignored)
```

---

## 4. Environment setup

Tested with Python 3.12.13, CUDA GPUs, PyTorch 2.8.0, scikit-learn 1.9.0,
sentence-transformers 4.1.0, transformers 4.57.6. vCache requires
`python >= 3.10, < 3.13`.

```bash
# 1. Create and activate a conda environment
conda create -n vcache python=3.12 -y
conda activate vcache

# 2. Install vCache (editable) with its dependencies from pyproject.toml
cd /path/to/vCache
pip install -e .

# 3. Install the extra dependencies of this package
pip install -r per_tuple_threshold/requirements.txt

# 4. Check the installation
python -c "import vcache, torch, sentence_transformers, anthropic; print(torch.cuda.is_available())"
python -m per_tuple_threshold --help
```

Notes:

- `pip install -e .` also installs vLLM (a vCache dependency), which pins its own
  PyTorch build. If PyTorch is installed first for a specific CUDA version, install
  it from the [PyTorch index](https://pytorch.org/get-started/locally/), e.g.
  `pip install torch --index-url https://download.pytorch.org/whl/cu128`.
- All commands below are run from the **repository root**, so that both `vcache`
  and `per_tuple_threshold` are importable.
- GPU memory: Qwen3-Embedding-8B in bfloat16 needs roughly 16 GB. The device with
  the most free memory is chosen automatically. Qwen3-Embedding-0.6B is a smaller
  alternative (results will differ).

---

## 5. External resources

### 5.1 SemBench dataset

The tuples are rows of `SemBench/files/movie/data/sf_2000/Reviews.csv`, which is
committed in the SemBench repository. Clone it **next to** this repository (the
default location), or point to it explicitly:

```bash
cd /path/to            # the parent directory of vCache/
git clone https://github.com/SemBench/SemBench.git
# -> /path/to/SemBench/files/movie/data/sf_2000/Reviews.csv

# Alternatively, for a checkout elsewhere:
export SEMBENCH_ROOT=/path/to/SemBench     # or pass --sembench-root
```


The base queries reproduce the semantic operators of SemBench movie queries
Q1-Q10 (`SemBench/files/movie/query/bigquery/Q*.sql`); see `predicates.py` for the
mapping. The sampled tuples are fixed in `per_tuple_threshold/data/sample_ids.json`.

### 5.2 LLM provider and API key

Query generation and labeling use one LLM, from one of three providers:

| Provider | API | Default model | Key | Endpoint |
|---|---|---|---|---|
| `anthropic` (default) | Messages API | `claude-sonnet-4-5-20250929` | `ANTHROPIC_AUTH_TOKEN` or `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL` (optional) |
| `openai` | Chat Completions API | `gpt-4o-mini` | `OPENAI_API_KEY` | `OPENAI_BASE_URL` (optional) |
| `local` | Chat Completions API on your own server (e.g. vLLM) | none: pass the served model name | `LLM_API_KEY` (optional) | `LLM_BASE_URL` (**required**) |

Keys are read **only from environment variables**; never put them in code or
commit them.

**Anthropic (Claude):**

```bash
export ANTHROPIC_AUTH_TOKEN="sk-..."                   # or ANTHROPIC_API_KEY
export ANTHROPIC_BASE_URL="https://api.camel-hub.com"  # optional gateway; omit for api.anthropic.com
python -m per_tuple_threshold calibrate --llm-model claude-sonnet-4-5-20250929
```

**OpenAI (GPT):**

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.camel-hub.com/v1"  # optional OpenAI-compatible gateway; omit for api.openai.com
python -m per_tuple_threshold calibrate --llm-provider openai --llm-model gpt-4o
```

**Local server (e.g. vLLM on a GPU node, no internet needed):**

```bash
vllm serve /path/to/Qwen2.5-14B-Instruct --served-model-name qwen2.5-14b --port 8000 &
export LLM_BASE_URL="http://localhost:8000/v1"        # selects the local provider
python -m per_tuple_threshold calibrate --llm-model qwen2.5-14b
```
`--llm-model` must equal the server's `--served-model-name`. No key is needed;
set `LLM_API_KEY` only if your server checks one.

How the provider is chosen: `--llm-provider`, then `LLM_PROVIDER`, then `local`
if `LLM_BASE_URL` is set, then the model name (`gpt-*`, `chatgpt-*`, `o1*`,
`o3*`, `o4*` are OpenAI; others are Anthropic). `--llm-model gpt-4.1` therefore
selects OpenAI on its own. Without a model, `LLM_MODEL` or the provider's
default model is used; `local` has no default and stops with an error. Through
an OpenAI-compatible gateway, `--llm-provider openai` also works with non-GPT
model names, e.g. `--llm-provider openai --llm-model claude-sonnet-4-5-20250929`.

Notes:

- Use a non-reasoning chat model with OpenAI (GPT-4o, GPT-4.1 families). Labels
  are requested with `max_tokens=16` and `temperature=0`. Reasoning models
  (o-series, GPT-5) reject these parameters or spend the token budget on reasoning.
  The same applies to `local`: serve an instruct model with thinking disabled
  (e.g. Qwen2.5-Instruct, Llama-3.1-Instruct), not a reasoning model.
- The label cache key is `sha256(model, kind, prompt)`. Changing the model
  therefore never reuses another model's answers. To compare models, run stage 1
  once per model and use a separate `--output` for each.
- The client lives in `per_tuple_threshold/llm/client.py`. It is created only when
  an LLM call is needed: a run whose q' pools and labels are all cached needs no
  key (see 6.3).

### 5.3 Embedding model

Default: [`Qwen/Qwen3-Embedding-8B`](https://huggingface.co/Qwen/Qwen3-Embedding-8B).
Download it once to a local directory:

```bash
pip install -U "huggingface_hub[cli]"
# export HF_ENDPOINT=https://hf-mirror.com      # optional mirror if huggingface.co is unreachable
huggingface-cli download Qwen/Qwen3-Embedding-8B --local-dir /path/to/models/Qwen3-Embedding-8B
# (newer huggingface_hub versions: `hf download Qwen/Qwen3-Embedding-8B --local-dir ...`)

export EMBEDDING_MODEL=/path/to/models/Qwen3-Embedding-8B   # or pass --embedding-model
```

If `EMBEDDING_MODEL` is a Hugging Face id instead of a directory,
sentence-transformers downloads it into the Hugging Face cache on first use. When
it is a local directory, the code sets `HF_HUB_OFFLINE=1` so no network access is
attempted.

---

## 6. Running the pipeline

The pipeline has two stages that share one run directory.

### 6.1 Stage 1: calibrate

Generates `q'` with the LLM, labels every tuple, and fits `t(d)` per tuple and a
pooled `t(q)`. Set the data and embedding paths once, then pick one LLM provider:

```bash
conda activate vcache
cd /path/to/vCache

export EMBEDDING_MODEL=/path/to/embedding/model/Qwen3-Embedding-8B
export SEMBENCH_ROOT=/path/to/SemBench
```

**Option A: Anthropic (Claude, default)**

```bash
export ANTHROPIC_AUTH_TOKEN="sk-..."                   # or ANTHROPIC_API_KEY
export ANTHROPIC_BASE_URL="https://api.camel-hub.com"  # optional; omit for api.anthropic.com

python -m per_tuple_threshold calibrate               # default model: claude-sonnet-4-5-20250929
# -> experiments/per_tuple_threshold/results/run_<YYYYmmdd_HHMMSS>/
```

**Option B: OpenAI (GPT)**

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.camel-hub.com/v1"  # optional; omit for api.openai.com

python -m per_tuple_threshold calibrate --llm-provider openai --llm-model gpt-4o
# A gpt-* model name alone also selects OpenAI:
python -m per_tuple_threshold calibrate --llm-model gpt-4.1
# Or configure it through the environment:
export LLM_PROVIDER=openai LLM_MODEL=gpt-4o
python -m per_tuple_threshold calibrate
```

Use a non-reasoning chat model for OpenAI (GPT-4o / GPT-4.1 families); see 5.2.
All providers write the same output files. The label cache is keyed by model
name, so give each model its own `--output` directory when comparing models.

Useful options:

```bash
# Only some predicates
python -m per_tuple_threshold calibrate --predicates sembench_positive extra_enthusiastic

# Fixed output directory; rerunning it resumes from its llm_cache.json
python -m per_tuple_threshold calibrate --output experiments/per_tuple_threshold/results/my_run
```

A full run of all 17 predicates issues 17 generation calls (more if a pool has to
be regenerated) and about 9.7k unique labeling prompts (9,676 in the 2026-10-23 run). Labels are flushed to `llm_cache.json` every 10 calls, so an
interrupted run resumes without repeating them when it is restarted with the same
`--output`. Note that q' pools are regenerated (and differ, temperature 0.7)
unless `--pools` is given.

### 6.2 Stage 2: predict

Trains `g_theta` on a calibration subset and evaluates predicted thresholds
(Section 2). No LLM calls:

```bash
python -m per_tuple_threshold predict --run-dir experiments/per_tuple_threshold/results/run_<timestamp>

# Options
python -m per_tuple_threshold predict --run-dir <dir> \
    --calibration-fraction 0.667 --margin-delta 0.1 --seed 0 \
    --hidden-dim 16 --epochs 400 --learning-rate 0.01 --weight-decay 0.01
```

The embedding model defaults to the one recorded in `run_config.json`.


### 6.3 Using the filter from code

```python
from per_tuple_threshold.serving import PerTupleSemanticFilter

thresholds = {"movie_a:1": 0.41, "movie_b:7": 0.0, "movie_c:3": None}   # t_hat(d)
semantic_filter = PerTupleSemanticFilter(thresholds, margin=0.05)
decision = semantic_filter.decide(similarity=0.62)                    # s(q, q')
decision.reuse      # tuples whose cached a_d is reused
decision.recompute  # tuples that need LLM(q', d)
```

---

## 7. Output files

```
<run_dir>/
├── run_config.json            stage-1 settings (paths, models, predicates)
├── generated_pools.json       q' candidates per predicate and category
├── llm_cache.json             sha256(model, kind, prompt) -> answer
├── calibration/
│   ├── <predicate>.json       predicate, selected variants with s(q, q'), per-tuple records
│   │                          (cached answer, t(d), gamma, var_t, every observation),
│   │                          pooled t(q), held-out scores (per_tuple vs. unified)
│   └── metrics.json           per-predicate summary without the per-tuple records
└── prediction/
    ├── <predicate>.json       split, features, LOO predictions, margin, t_safe per target
    │                          tuple, held-out scores of the four rules, cost
    └── metrics.json           per-predicate summary and the stage-2 configuration
```

A held-out score is

```json
{"decisions": 96, "reuses": 91, "correct_reuses": 90, "false_reuses": 1,
 "correct_rejects": 1, "precision": 0.989, "decision_accuracy": 0.948}
```

where a decision is one `(q', d)` pair, a reuse is correct when
`LLM(q', d) == a_d`, and a reject is correct when it differs. Scores are reported
for the groups `paraphrase`, `unrelated` and `all`. Both stages also print a
short summary table to the terminal.

---

## 8. Configuration reference

| Setting | CLI flag | Environment variable | Default |
|---|---|---|---|
| SemBench checkout | `--sembench-root` | `SEMBENCH_ROOT` | `../SemBench` (sibling of this repo) |
| Embedding model | `--embedding-model` | `EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-8B` |
| LLM provider | `--llm-provider` | `LLM_PROVIDER` | `local` if `LLM_BASE_URL` is set, else inferred from the model name, else `anthropic` |
| LLM model | `--llm-model` | `LLM_MODEL` | `claude-sonnet-4-5-20250929` (anthropic), `gpt-4o-mini` (openai), none (local) |
| Anthropic API key | - | `ANTHROPIC_AUTH_TOKEN` or `ANTHROPIC_API_KEY` | required for uncached Anthropic calls |
| Anthropic endpoint | - | `ANTHROPIC_BASE_URL` | `api.anthropic.com` |
| OpenAI API key | - | `OPENAI_API_KEY` | required for uncached OpenAI calls |
| OpenAI endpoint | - | `OPENAI_BASE_URL` | `api.openai.com` |
| Local server endpoint | - | `LLM_BASE_URL` | required for `local` (e.g. `http://localhost:8000/v1`) |
| Local server key | - | `LLM_API_KEY` | optional; a placeholder is sent when unset |
| Output directory | `--output` | - | `experiments/per_tuple_threshold/results/run_<timestamp>` |
| Parallel LLM requests | `--workers` | - | 8 |

Algorithm constants live next to the code that uses them: similarity targets in
`calibration/variant_selection.py`, generation settings in
`calibration/query_generation.py`, the minimum of 6 observations in
`calibration/threshold.py`, the margin cap in `prediction/margin.py`.

---

## 9. Implementation notes and limitations

- **Similarity** is the cosine between the embeddings of the two query strings,
  both encoded with Qwen3's `query` prompt. The tuple is not part of the vector;
  per-tuple behaviour enters only through `t(d)`.
- **Estimator.** `t(d)` is vCache's estimate of the point where
  `P(match | s) = 0.5`. The online vCache policy additionally uses `var_t` and
  `delta` to build a confidence bound before exploiting; this package stores
  `var_t` but the reuse rule is `s >= t(d)` (plus the margin for predicted
  thresholds). The margin is an empirical leave-one-out quantile, not a formal
  guarantee.
- **vCache change.** The only edit to `vcache/` used here is
  `LogisticRegression(penalty=None, ...)` -> `LogisticRegression(C=np.inf, ...)`
  in `verified.py`, both of which fit an unpenalised model, for compatibility with
  scikit-learn 1.9.
- **Sample size.** The frozen sample has 24, 12, 12 and 16 tuples per population,
  so stage 2 trains each `g_theta` on at most 16 points. The stage demonstrates
  the method end to end; its numbers are not statistically conclusive.
- **Duplicates.** The `ant_man` population lists two reviews twice. They count
  twice in pooled fits; the stage-2 split keeps each duplicate on one side.
- **Determinism.** Labels use temperature 0 and are cached, and q' generation
  uses temperature 0.7. Re-running with `--pools` and `--label-cache` reproduces a
  run exactly.

---

## 10. Tests

```bash
python -m pytest tests/test_per_tuple_threshold -q
```

The tests cover the vCache estimator wrapper, the adjective-swap filter, variant
selection, label parsing, the serving rule, the margin, and the data split. They
need neither a GPU nor network access.

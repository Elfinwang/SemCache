"""Default locations and model identifiers.

Resolution order for every setting: command-line flag, then environment
variable, then the default defined here.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parent

RESULTS_DIR = REPO_ROOT / "experiments" / "per_tuple_threshold" / "results"
SAMPLE_IDS_PATH = PACKAGE_ROOT / "data" / "sample_ids.json"

SEMBENCH_ROOT_ENV = "SEMBENCH_ROOT"
EMBEDDING_MODEL_ENV = "EMBEDDING_MODEL"
LLM_MODEL_ENV = "LLM_MODEL"
LLM_PROVIDER_ENV = "LLM_PROVIDER"

# SemBench is expected as a sibling checkout of this repository.
DEFAULT_SEMBENCH_ROOT = REPO_ROOT.parent / "SemBench"
REVIEWS_RELATIVE_PATH = Path("files/movie/data/sf_2000/Reviews.csv")

# A Hugging Face model id or a local directory with the downloaded weights.
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"

# Used both to generate the query variants q' and to label LLM(q', d).
DEFAULT_LLM_PROVIDER = "anthropic"
DEFAULT_LLM_MODELS = {
    "anthropic": "claude-sonnet-4-5-20250929",
    "openai": "gpt-4o-mini",
}


def default_sembench_root() -> Path:
    return Path(os.environ.get(SEMBENCH_ROOT_ENV, DEFAULT_SEMBENCH_ROOT))


def default_embedding_model() -> str:
    return os.environ.get(EMBEDDING_MODEL_ENV, DEFAULT_EMBEDDING_MODEL)


def resolve_llm(model: str | None, provider: str | None) -> tuple[str, str]:
    """Return ``(provider, model)``.

    The provider comes from the flag, then ``LLM_PROVIDER``, then the model
    name (``gpt-*`` and ``o*`` models are OpenAI), then the default. A missing
    model falls back to ``LLM_MODEL``, then the provider's default model.
    """
    from per_tuple_threshold.llm.client import infer_provider

    model = model or os.environ.get(LLM_MODEL_ENV)
    provider = provider or os.environ.get(LLM_PROVIDER_ENV)
    if provider is None:
        provider = infer_provider(model) if model else DEFAULT_LLM_PROVIDER
    if provider not in DEFAULT_LLM_MODELS:
        raise ValueError(f"Unknown LLM provider {provider!r}; choose from {sorted(DEFAULT_LLM_MODELS)}.")
    return provider, model or DEFAULT_LLM_MODELS[provider]


def reviews_path(sembench_root: Path) -> Path:
    return Path(sembench_root) / REVIEWS_RELATIVE_PATH

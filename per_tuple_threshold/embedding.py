"""Sentence embeddings and the similarity s(q, q').

s(q, q') is the cosine of the L2-normalised embeddings of the two query
strings. The tuple is not part of either vector. Queries use the model's
``query`` prompt; tuple texts (used only as features) are encoded without one.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

BATCH_SIZE = 16


def configure_offline_mode(model: str) -> None:
    """Disable Hub lookups for a local model directory.

    ``huggingface_hub`` reads these variables at import time, so this must run
    before ``transformers``, ``sentence_transformers`` or ``vcache`` is imported.
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if Path(model).is_dir():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def pick_device() -> str:
    """The CUDA device with the most free memory, else CPU."""
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    free = [(torch.cuda.mem_get_info(index)[0], index) for index in range(torch.cuda.device_count())]
    return f"cuda:{max(free)[1]}"


class Embedder:
    def __init__(self, model: str, device: str | None = None):
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_name = model
        self.device = device or pick_device()
        logger.info("Loading embedding model %s on %s", model, self.device)
        self.model = SentenceTransformer(
            model,
            device=self.device,
            model_kwargs={"torch_dtype": torch.bfloat16},
            tokenizer_kwargs={"padding_side": "left"},
        )
        self._has_query_prompt = "query" in (self.model.prompts or {})

    def encode_queries(self, texts: list[str]) -> dict[str, np.ndarray]:
        prompt_name = "query" if self._has_query_prompt else None
        return self._encode(texts, prompt_name)

    def encode_documents(self, texts: list[str]) -> dict[str, np.ndarray]:
        return self._encode(texts, None)

    def _encode(self, texts: list[str], prompt_name: str | None) -> dict[str, np.ndarray]:
        # Batch composition changes bf16 results slightly; a fixed order keeps
        # similarities independent of the order texts are passed in.
        unique = sorted(set(texts))
        vectors = self.model.encode(
            unique,
            prompt_name=prompt_name,
            normalize_embeddings=True,
            batch_size=BATCH_SIZE,
            show_progress_bar=len(unique) > BATCH_SIZE,
        )
        return {text: np.asarray(vector, dtype=np.float64) for text, vector in zip(unique, vectors)}


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    """Cosine similarity of two L2-normalised vectors."""
    return float(np.dot(left, right))

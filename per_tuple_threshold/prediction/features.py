"""Observable tuple features phi(d). Computing them costs no LLM call.

Features, in order of ``FEATURE_NAMES``:

* ``log_word_count``: log(1 + words) / log(81).
* ``exclamations``: min(count of "!", 3) / 3.
* ``lexicon_positive`` / ``lexicon_negative``: share of words in a small
  sentiment lexicon.
* ``sim_enthusiastic`` / ``sim_mild`` / ``sim_negative``: cosine between the
  tuple embedding and three anchor reviews.
* ``cached_answer``: the already-paid answer a_d = LLM(q, d); 1/0 for YES/NO,
  (score - 1) / 4 for 1-5 scores.
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np

from per_tuple_threshold.embedding import Embedder
from per_tuple_threshold.schema import KIND_SCORE

FEATURE_NAMES = (
    "log_word_count",
    "exclamations",
    "lexicon_positive",
    "lexicon_negative",
    "sim_enthusiastic",
    "sim_mild",
    "sim_negative",
    "cached_answer",
)

ANCHORS = {
    "enthusiastic": "Absolutely love it. Best movie ever. Brilliant, thrilling, a masterpiece.",
    "mild": "It's okay. Works fine. Nothing special, not terrible.",
    "negative": "Terrible. Completely disappointing. A dull waste of time.",
}

POSITIVE_WORDS = frozenset({
    "love", "loved", "amazing", "brilliant", "excellent", "great", "wonderful",
    "outstanding", "superb", "fantastic", "masterpiece", "delight", "delightful",
    "hilarious", "perfect", "beautiful", "gripping", "compelling", "witty",
    "charming", "thrilling", "enjoyable", "fun", "best", "recommend",
})
NEGATIVE_WORDS = frozenset({
    "bad", "worse", "worst", "boring", "dull", "terrible", "awful", "hate",
    "hated", "waste", "poor", "disappointing", "disappointment", "mess",
    "mediocre", "bland", "stupid", "unfunny", "painful", "avoid", "weak",
    "failure", "fail", "lifeless", "hollow",
})

_WORD_RE = re.compile(r"[a-zA-Z']+")
_WORD_COUNT_SCALE = np.log1p(80.0)
_MAX_EXCLAMATIONS = 3


def lexicon_rates(text: str) -> tuple[float, float]:
    words = [word.lower() for word in _WORD_RE.findall(text)]
    if not words:
        return 0.0, 0.0
    positive = sum(word in POSITIVE_WORDS for word in words) / len(words)
    negative = sum(word in NEGATIVE_WORDS for word in words) / len(words)
    return positive, negative


def encode_cached_answer(answer: str, kind: str) -> float:
    if kind == KIND_SCORE:
        return (float(answer) - 1.0) / 4.0
    return 1.0 if answer == "YES" else 0.0


class FeatureExtractor:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        vectors = embedder.encode_documents(list(ANCHORS.values()))
        self._anchors = np.stack([vectors[text] for text in ANCHORS.values()])

    def transform(self, texts: Sequence[str], cached_answers: Sequence[str], kind: str) -> np.ndarray:
        vectors = self.embedder.encode_documents(list(texts))
        rows = []
        for text, answer in zip(texts, cached_answers):
            positive, negative = lexicon_rates(text)
            anchor_sims = self._anchors @ vectors[text]
            rows.append([
                np.log1p(len(_WORD_RE.findall(text))) / _WORD_COUNT_SCALE,
                min(text.count("!"), _MAX_EXCLAMATIONS) / _MAX_EXCLAMATIONS,
                positive,
                negative,
                *anchor_sims.tolist(),
                encode_cached_answer(answer, kind),
            ])
        return np.asarray(rows, dtype=np.float32)

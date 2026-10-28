"""Observable features phi(d) for TPC-H product reviews. No LLM call.

Same eight features, in the same order, as ``prediction.features`` so
results stay comparable; only the anchors and the lexicon are product-review
specific. ``TpchFeatureExtractor`` has the ``transform`` interface that
``pipeline.predict.predict_predicate`` uses.
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np

from per_tuple_threshold.prediction.features import encode_cached_answer

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
    "enthusiastic": "Excellent part. It arrived early, works perfectly, and the finish is flawless. Highly recommend this supplier.",
    "mild": "The part is okay. Delivery was a little slow, but it does the job.",
    "negative": "Terrible order. It arrived weeks late, did not match our spec, and we sent it back.",
}

POSITIVE_WORDS = frozenset({
    "excellent", "great", "perfect", "perfectly", "flawless", "quick", "fast", "early",
    "prompt", "promptly", "reliable", "sturdy", "solid", "recommend", "pleased", "satisfied",
    "happy", "smooth", "impressed", "impressive", "efficient", "quality", "best",
})
NEGATIVE_WORDS = frozenset({
    "late", "delay", "delayed", "delays", "returned", "return", "defective", "broken",
    "damaged", "disappointed", "disappointing", "poor", "wrong", "slow", "frustrating",
    "frustrated", "unacceptable", "worst", "issue", "issues", "problem", "problems", "concerns",
})

_WORD_RE = re.compile(r"[a-zA-Z']+")
# Generated reviews run 40-90 words, longer than SemBench critic quotes.
_WORD_COUNT_SCALE = np.log1p(120.0)
_MAX_EXCLAMATIONS = 3


def lexicon_rates(text: str) -> tuple[float, float]:
    words = [word.lower() for word in _WORD_RE.findall(text)]
    if not words:
        return 0.0, 0.0
    positive = sum(word in POSITIVE_WORDS for word in words) / len(words)
    negative = sum(word in NEGATIVE_WORDS for word in words) / len(words)
    return positive, negative


class TpchFeatureExtractor:
    def __init__(self, embedder):
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

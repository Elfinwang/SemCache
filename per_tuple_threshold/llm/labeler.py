"""Cached LLM answers for (query, tuple) prompts.

A reuse is correct when ``LLM(q', d)`` equals the cached ``a_d = LLM(q, d)``
as a string. Yes/no queries are answered with ``YES`` or ``NO``; score
queries with one integer.

The cache key is ``sha256(model, kind, prompt)``, so cache files are portable
across runs and the client is only created on a cache miss.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Hashable, Iterable

from per_tuple_threshold.llm.client import make_client
from per_tuple_threshold.schema import KIND_SCORE

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 6
FLUSH_EVERY = 10

LabelJob = tuple[Hashable, str, str]


def build_prompt(kind: str, question: str, context: str) -> str:
    if kind == KIND_SCORE:
        return f"{question}\n{context}\nReply with a single integer from 1 to 5 only."
    return f"{question}\n{context}\nReply with YES or NO only."


def parse_label(text: str, kind: str) -> str:
    raw = text.strip()
    token = raw.upper()
    if kind == KIND_SCORE:
        # A reply that is a bare integer is kept verbatim, even outside 1-5.
        if raw.lstrip("-").isdigit():
            return raw
        for char in token:
            if char in "12345":
                return char
        raise ValueError(text)
    if token.startswith("YES"):
        return "YES"
    if token.startswith("NO"):
        return "NO"
    raise ValueError(text)


def _reminder(kind: str) -> str:
    if kind == KIND_SCORE:
        return "Reply with a single integer from 1 to 5 only."
    return "Reply with YES or NO only. If the review does not establish the claim, answer NO."


class CachedLabeler:
    """Thread-safe labeler backed by a JSON cache file."""

    def __init__(self, provider: str, model: str, cache_path: Path):
        self.provider = provider
        self.model = model
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, str] = (
            json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        )
        self.api_calls = 0
        self._client = None
        self._lock = threading.Lock()

    @property
    def client(self):
        with self._lock:
            if self._client is None:
                self._client = make_client(self.provider, self.model)
            return self._client

    def cache_key(self, prompt: str, kind: str) -> str:
        return hashlib.sha256(f"{self.model}\n{kind}\n{prompt}".encode()).hexdigest()

    def label(self, prompt: str, kind: str) -> str:
        key = self.cache_key(prompt, kind)
        with self._lock:
            cached = self.cache.get(key)
        if cached is not None:
            return cached
        answer = self._request(prompt, kind)
        with self._lock:
            self.cache[key] = answer
            self.api_calls += 1
            if self.api_calls % FLUSH_EVERY == 0:
                self._write()
        return answer

    def flush(self) -> None:
        with self._lock:
            self._write()

    def _write(self) -> None:
        self.cache_path.write_text(json.dumps(self.cache))

    def _request(self, prompt: str, kind: str) -> str:
        client = self.client
        messages = [{"role": "user", "content": prompt}]
        last_error: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                text = client.complete(messages, max_tokens=16, temperature=0)
                return parse_label(text, kind)
            except ValueError as exc:
                last_error = exc
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": str(exc)[:500]},
                    {"role": "user", "content": _reminder(kind)},
                ]
                time.sleep(1)
            except Exception as exc:
                last_error = exc
                delay = 2**attempt
                if "429" in str(exc) or "rate" in str(exc).lower():
                    delay = max(delay, 8)
                time.sleep(delay)
        raise RuntimeError(f"LLM labeling failed after {MAX_ATTEMPTS} attempts: {last_error}") from last_error


def label_all(jobs: Iterable[LabelJob], labeler: CachedLabeler, workers: int) -> dict[Hashable, str]:
    """Label ``(key, prompt, kind)`` jobs in parallel, de-duplicated by key."""
    unique: dict[Hashable, tuple[str, str]] = {}
    for key, prompt, kind in jobs:
        unique.setdefault(key, (prompt, kind))
    logger.info("Labeling %d unique prompts (%d cached entries)", len(unique), len(labeler.cache))
    answers: dict[Hashable, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(labeler.label, prompt, kind): key
            for key, (prompt, kind) in unique.items()
        }
        for done, future in enumerate(as_completed(futures), start=1):
            answers[futures[future]] = future.result()
            if done % 200 == 0 or done == len(futures):
                logger.info("Labeled %d/%d, new API calls %d", done, len(futures), labeler.api_calls)
    labeler.flush()
    return answers

"""LLM-generated q' pools for TPC-H product-review predicates.

Mirrors ``calibration.query_generation`` (same JSON keys, counts, temperature,
adjective-swap filter and minimum pool sizes) with product-review wording and
examples. The core module is not modified: its public ``clean_pool``,
``parse_json_block`` and ``MIN_POOL_SIZE`` are reused, and the small retry
helpers are re-implemented here.
"""

from __future__ import annotations

import json
import logging

from per_tuple_threshold.calibration.query_generation import (
    GENERATION_MAX_TOKENS,
    GENERATION_TEMPERATURE,
    MAX_ATTEMPTS,
    MIN_POOL_SIZE,
    Pool,
    clean_pool,
    parse_json_block,
)
from per_tuple_threshold.llm.client import ChatClient
from per_tuple_threshold.schema import KIND_SCORE, Predicate

logger = logging.getLogger(__name__)


def generation_prompt(predicate: Predicate) -> str:
    if predicate.kind == KIND_SCORE:
        answer_form = (
            "Every query must still be answered by one integer from 1 to 5. "
            "Say what 1 and 5 mean inside the query."
        )
    else:
        answer_form = "Every query must be a yes/no question."
    return f"""Generate calibration queries for a semantic cache.

Base query q:
{predicate.base}

The user will supply one customer review of an industrial part. {answer_form}

Return one JSON object and nothing else, with these arrays of strings:
- "paraphrases": 16 queries with the same meaning as q and different wording.
- "intensity": 8 queries about the same attribute at a weaker or stronger degree.
- "should_reuse": 6 queries whose correct decision is to reuse a cached answer to q.
- "should_not_reuse": 8 queries whose correct decision is to not reuse a cached answer to q. Change the information need. Do not only insert "not" or swap a single polarity word.
- "unrelated": 8 queries about a completely different fact in the same review.

Do not repeat q. Do not repeat a query across arrays.

Never keep q's sentence and only replace a word with its opposite.
Forbidden pattern: q is "Determine if the following product review is clearly positive."
and a candidate is "Determine if the following product review is clearly negative."
An opposite query must use different syntax and different content words, not an adjective swap.
Also forbidden: "Does this review say the delivery was fast?" rewritten as "Does this review say the delivery was slow?"
"""


def _rejection_note(rejected: list[str]) -> str:
    return (
        "\n\nThese were rejected because they only flip a polarity word. "
        "Do not write anything in this pattern:\n"
        + "\n".join(f"- {text}" for text in rejected)
    )


def _is_sufficient(pool: Pool) -> bool:
    return all(len(pool[category]) >= size for category, size in MIN_POOL_SIZE.items())


def generate_pool(predicate: Predicate, client: ChatClient) -> Pool:
    base_prompt = generation_prompt(predicate)
    prompt = base_prompt
    last_error: Exception | str | None = None
    for _ in range(MAX_ATTEMPTS):
        text = client.complete(
            [{"role": "user", "content": prompt}],
            max_tokens=GENERATION_MAX_TOKENS,
            temperature=GENERATION_TEMPERATURE,
        )
        try:
            pool, rejected = clean_pool(predicate.base, parse_json_block(text))
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            continue
        if rejected:
            logger.info("%s: dropped %d adjective-swap candidates", predicate.id, len(rejected))
            prompt = base_prompt + _rejection_note(rejected)
        if _is_sufficient(pool):
            return pool
        last_error = f"pool too small: { {key: len(value) for key, value in pool.items()} }"
    raise RuntimeError(f"Query generation failed for {predicate.id}: {last_error}")

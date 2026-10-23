"""LLM-generated candidate pools of query variants q'.

For one base query q the LLM writes five kinds of candidates:

* ``paraphrase``: same meaning, different wording.
* ``intensity``: the same attribute at a weaker or stronger degree.
* ``should_reuse``: queries whose cached answer to q should be reused.
* ``should_not_reuse``: queries with a different information need.
* ``unrelated``: queries about a different fact of the same tuple.

Candidates that only flip a polarity word of q ("clearly positive" ->
"clearly negative") are discarded, and the generator is asked again with the
rejected strings as negative examples. ``variant_selection`` later picks the
calibration set from these pools by embedding similarity.
"""

from __future__ import annotations

import json
import logging
import re

from per_tuple_threshold.llm.client import ChatClient
from per_tuple_threshold.schema import KIND_SCORE, Predicate

logger = logging.getLogger(__name__)

# JSON key in the LLM reply -> pool category.
RESPONSE_KEYS = {
    "paraphrases": "paraphrase",
    "intensity": "intensity",
    "should_reuse": "should_reuse",
    "should_not_reuse": "should_not_reuse",
    "unrelated": "unrelated",
}
CATEGORIES = tuple(RESPONSE_KEYS.values())

# Minimum pool sizes needed by variant_selection.
MIN_POOL_SIZE = {"paraphrase": 8, "unrelated": 4, "should_not_reuse": 4}

GENERATION_TEMPERATURE = 0.7
GENERATION_MAX_TOKENS = 4096
MAX_ATTEMPTS = 3

POLARITY_WORDS = frozenset({
    "positive", "negative", "positivity", "negativity", "favorable", "unfavorable",
    "favourable", "unfavourable", "good", "bad", "praise", "criticize", "criticise",
    "praising", "criticizing", "love", "hate", "liked", "disliked", "like", "dislike",
    "enthusiastic", "bored", "enthusiasm", "boredom", "funny", "boring", "recommend",
    "skip", "excellent", "poor", "same", "opposite", "delighted", "disappointed",
    "yes", "no", "warm", "cold", "approve", "disapprove",
})
NEGATION_WORDS = frozenset({"not", "no", "never", "nt"})
ANTONYM_PAIRS = frozenset(
    frozenset(pair)
    for pair in (
        ("positive", "negative"), ("good", "bad"), ("love", "hate"),
        ("praise", "criticize"), ("praise", "criticise"), ("funny", "scary"),
        ("funny", "boring"), ("funny", "unfunny"), ("same", "opposite"),
        ("enthusiastic", "bored"), ("like", "dislike"), ("yes", "no"),
        ("favorable", "unfavorable"), ("delighted", "disappointed"),
        ("warm", "cold"), ("approve", "disapprove"), ("excellent", "poor"),
        ("recommend", "avoid"), ("recommend", "skip"),
    )
)

Pool = dict[str, list[str]]


def _tokens(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


def is_adjective_swap(base: str, candidate: str, category: str | None = None) -> bool:
    """True when ``candidate`` keeps the sentence of ``base`` and only flips or negates a word."""
    base_tokens = _tokens(base)
    candidate_tokens = _tokens(candidate)
    if not base_tokens or not candidate_tokens:
        return False
    only_base = set(base_tokens) - set(candidate_tokens)
    only_candidate = set(candidate_tokens) - set(base_tokens)
    if only_base and only_candidate and only_base <= POLARITY_WORDS and only_candidate <= POLARITY_WORDS:
        return True
    if (
        (only_base or only_candidate)
        and (only_base | only_candidate) <= NEGATION_WORDS
        and set(base_tokens) - NEGATION_WORDS == set(candidate_tokens) - NEGATION_WORDS
    ):
        return True
    if len(base_tokens) == len(candidate_tokens):
        changed = [(left, right) for left, right in zip(base_tokens, candidate_tokens) if left != right]
        if len(changed) == 1:
            left, right = changed[0]
            if frozenset({left, right}) in ANTONYM_PAIRS or category == "should_not_reuse":
                return True
    return False


def generation_prompt(predicate: Predicate) -> str:
    if predicate.kind == KIND_SCORE:
        answer_form = (
            "Every query must still be answered by one integer from 1 to 5. "
            "Say what 1 and 5 mean inside the query."
        )
    else:
        answer_form = "Every query must be a yes/no question."
    if predicate.population == "ant_man_pairs":
        data_form = "The user will supply two movie reviews."
    else:
        data_form = "The user will supply one movie review."
    return f"""Generate calibration queries for a semantic cache.

Base query q:
{predicate.base}

{data_form} {answer_form}

Return one JSON object and nothing else, with these arrays of strings:
- "paraphrases": 16 queries with the same meaning as q and different wording.
- "intensity": 8 queries about the same attribute at a weaker or stronger degree.
- "should_reuse": 6 queries whose correct decision is to reuse a cached answer to q.
- "should_not_reuse": 8 queries whose correct decision is to not reuse a cached answer to q. Change the information need. Do not only insert "not" or swap a single polarity word.
- "unrelated": 8 queries about a completely different fact in the same review or reviews.

Do not repeat q. Do not repeat a query across arrays.

Never keep q's sentence and only replace a word with its opposite.
Forbidden pattern: q is "Determine if the following movie review is clearly positive."
and a candidate is "Determine if the following movie review is clearly negative."
An opposite query must use different syntax and different content words, not an adjective swap.
Also forbidden: "Does this review say the movie is funny?" rewritten as "Does this review say the movie is scary?"
"""


def _rejection_note(rejected: list[str]) -> str:
    return (
        "\n\nThese were rejected because they only flip a polarity word. "
        "Do not write anything in this pattern:\n"
        + "\n".join(f"- {text}" for text in rejected)
    )


def parse_json_block(text: str) -> dict:
    raw = text.strip()
    if "```" in raw:
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
    start, end = raw.find("{"), raw.rfind("}")
    return json.loads(raw[start : end + 1])


def clean_pool(base: str, parsed: dict) -> tuple[Pool, list[str]]:
    """Normalise whitespace, drop duplicates and adjective swaps."""
    seen = {base.strip()}
    pool: Pool = {}
    rejected: list[str] = []
    for response_key, category in RESPONSE_KEYS.items():
        kept = []
        for text in parsed.get(response_key) or []:
            text = " ".join(str(text).split())
            if not text or text in seen:
                continue
            if is_adjective_swap(base, text, category):
                rejected.append(text)
                continue
            seen.add(text)
            kept.append(text)
        pool[category] = kept
    return pool, rejected


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

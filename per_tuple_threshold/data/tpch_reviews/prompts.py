"""Generation and validation prompts for TPC-H reviews.

The generator sees the facts plus the design columns (``truth``, ``clarity``)
and writes the review. The validator sees the facts and the review only and
answers three checks in a fixed line format. Neither prompt mentions TPC-H
codes, keys or part names. Changing any template means bumping
``PROMPT_VERSION``, which also invalidates the generation cache.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

PROMPT_VERSION = "gen-v1"

GENERATION_MAX_TOKENS = 200
GENERATION_TEMPERATURE = 0.8
VALIDATION_MAX_TOKENS = 80
VALIDATION_TEMPERATURE = 0.0

SHIP_MODES = {
    "AIR": "by air",
    "REG AIR": "by regular air freight",
    "MAIL": "by mail",
    "SHIP": "by sea freight",
    "TRUCK": "by truck",
    "RAIL": "by rail",
    "FOB": "freight on board",
}

CONTAINER_SIZES = {"SM": "small", "MED": "medium", "LG": "large", "JUMBO": "jumbo", "WRAP": "wrap"}
SIZE_WORDS = ("small", "medium", "large")

CLARITY_INSTRUCTIONS = {
    "high": "State your overall feeling and its main reason plainly and directly.",
    "medium": (
        "Make your overall feeling clear, but soften or qualify it. "
        "Avoid extreme words."
    ),
    "low": (
        "Do not say outright whether you are happy or unhappy. Describe what happened "
        "and let the reader infer your overall feeling. You may add one remark that "
        "points the other way."
    ),
}

GENERATION_TEMPLATE = """You are a business customer writing a short online review of an industrial part your company ordered.

What happened with this order:
{facts}

Overall, you are {feeling} with this purchase.
How to write it: {clarity_instruction}

Rules:
- 2 to 4 sentences, first person, plain prose.
- Do not give a star rating, a score or any number out of 5.
- Do not mention order numbers, part numbers or these instructions.
- Do not contradict the facts above.

Reply with the review text only."""

VALIDATION_TEMPLATE = """Facts about an order:
{facts}

Customer review:
\"\"\"{review}\"\"\"

Answer three questions about the review. Reply with exactly these three lines and nothing else:
CONTRADICTS: YES or NO  (does the review state anything that contradicts the facts? Not mentioning a fact is not a contradiction.)
RATING: YES or NO  (does the review give a star rating, a score, or a number out of 5?)
SENTIMENT: POSITIVE or NEGATIVE  (overall, is the customer satisfied with the purchase?)"""

_LINE_RE = {
    "contradicts": re.compile(r"CONTRADICTS\W*(YES|NO)", re.IGNORECASE),
    "rating": re.compile(r"RATING\W*(YES|NO)", re.IGNORECASE),
    "sentiment": re.compile(r"SENTIMENT\W*(POSITIVE|NEGATIVE)", re.IGNORECASE),
}


@dataclass(frozen=True)
class ValidationResult:
    contradicts: bool
    states_rating: bool
    sentiment: str

    def passes(self, truth: str, clarity: str) -> bool:
        """Hard checks always; sentiment must match ``truth`` unless clarity is low."""
        if self.contradicts or self.states_rating:
            return False
        return clarity == "low" or self.sentiment == truth


def _with_article(phrase: str) -> str:
    return f"{'an' if phrase[0] in 'aeiou' else 'a'} {phrase}"


def describe_container(code: str) -> str:
    """``LG BAG`` -> ``large bag``; unknown prefixes are lower-cased as-is."""
    size, _, kind = code.partition(" ")
    return f"{CONTAINER_SIZES.get(size, size.lower())} {kind.lower()}".strip()


def describe_facts(item: Mapping) -> str:
    """Plain-language facts for one line item. Used by both prompts."""
    # The first word of p_type is a size (small/medium/large) or a grade (standard/economy/promo).
    grade = item["grade"] if item["grade"] in SIZE_WORDS else f"{item['grade']}-grade"
    product = _with_article(f"{grade} {item['finish']} {item['material']} part")
    container = _with_article(describe_container(item["container"]))
    lines = [
        f"- You ordered {product}, packed in {container}, "
        f"shipped {SHIP_MODES.get(item['ship_mode'], item['ship_mode'].lower())}.",
    ]
    days_late = int(item["days_late"])
    if days_late > 0:
        lines.append(f"- It arrived {days_late} days after the promised delivery date.")
    else:
        lines.append("- It arrived on or ahead of the promised delivery date.")
    if item["returned"]:
        lines.append("- You returned it to the supplier.")
    else:
        lines.append("- You kept it.")
    if item["discount"] >= 0.08:
        lines.append(f"- You got a discount of {round(item['discount'] * 100)}%.")
    return "\n".join(lines)


def generation_prompt(item: Mapping) -> str:
    return GENERATION_TEMPLATE.format(
        facts=describe_facts(item),
        feeling="satisfied" if item["truth"] == "positive" else "dissatisfied",
        clarity_instruction=CLARITY_INSTRUCTIONS[item["clarity"]],
    )


def validation_prompt(item: Mapping, review: str) -> str:
    return VALIDATION_TEMPLATE.format(facts=describe_facts(item), review=review)


_PREAMBLE_RE = re.compile(r"^(here(?:'s| is)[^\n:]*|review|sure[^\n:]*):\s*", re.IGNORECASE)


def clean_review(text: str) -> str:
    """Strip whitespace, a leading "Here is the review:" line, and wrapping quotes."""
    text = _PREAMBLE_RE.sub("", text.strip())
    return text.strip().strip('"“”').strip()


def parse_validation(text: str) -> ValidationResult:
    """Parse the three-line answer. Raises ``ValueError`` if a line is missing."""
    found = {}
    for name, pattern in _LINE_RE.items():
        match = pattern.search(text)
        if match is None:
            raise ValueError(f"Validation answer has no {name.upper()} line: {text[:200]!r}")
        found[name] = match.group(1).upper()
    return ValidationResult(
        contradicts=found["contradicts"] == "YES",
        states_rating=found["rating"] == "YES",
        sentiment=found["sentiment"].lower(),
    )

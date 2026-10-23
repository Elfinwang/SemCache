"""Base queries q on the SemBench movie scenario.

The first five reproduce the semantic operators of SemBench movie Q1-Q10
(``files/movie/query/bigquery/Q*.sql``). The ``extra_*`` queries are additional
predicates over the same reviews. Calibration variants q' are not listed here;
they are generated per base query by ``calibration.query_generation``.
"""

from __future__ import annotations

from per_tuple_threshold.schema import KIND_SCORE, KIND_YES_NO, Predicate

PREDICATES: tuple[Predicate, ...] = (
    Predicate(
        id="sembench_positive",
        kind=KIND_YES_NO,
        population="general",
        covers=("Q1", "Q3", "Q4", "Q8"),
        base="Determine if the following movie review is clearly positive.",
    ),
    Predicate(
        id="sembench_positive_taken3",
        kind=KIND_YES_NO,
        population="taken_3",
        covers=("Q2",),
        base="Determine if the following movie review is clearly positive.",
    ),
    Predicate(
        id="sembench_same_sentiment",
        kind=KIND_YES_NO,
        population="ant_man_pairs",
        covers=("Q5",),
        base=(
            "These two movie reviews express the same sentiment - "
            "either both are positive or both are negative."
        ),
    ),
    Predicate(
        id="sembench_opposite_sentiment",
        kind=KIND_YES_NO,
        population="ant_man_pairs",
        covers=("Q6", "Q7"),
        base=(
            "These two movie reviews express opposite sentiments - "
            "one is positive and the other is negative."
        ),
    ),
    Predicate(
        id="sembench_score",
        kind=KIND_SCORE,
        population="ant_man",
        covers=("Q9", "Q10"),
        base=(
            "Score from 1 to 5 how much the reviewer liked the movie. "
            "5 very positive, 4 positive, 3 neutral, 2 negative, 1 very negative."
        ),
    ),
    Predicate(
        id="extra_negative",
        kind=KIND_YES_NO,
        population="general",
        base="Determine if the following movie review is clearly negative.",
    ),
    Predicate(
        id="extra_enthusiastic",
        kind=KIND_YES_NO,
        population="general",
        base="Determine if the reviewer is enthusiastic about the movie, not merely mildly positive.",
    ),
    Predicate(
        id="extra_recommend",
        kind=KIND_YES_NO,
        population="general",
        base="Would the reviewer recommend this movie to other people?",
    ),
    Predicate(
        id="extra_acting",
        kind=KIND_YES_NO,
        population="general",
        base="Does this review praise the acting?",
    ),
    Predicate(
        id="extra_humor",
        kind=KIND_YES_NO,
        population="general",
        base="Does this review say the movie is funny?",
    ),
    Predicate(
        id="extra_mixed",
        kind=KIND_YES_NO,
        population="general",
        base="Is this review mixed, with both positive and negative points?",
    ),
    Predicate(
        id="extra_family",
        kind=KIND_YES_NO,
        population="general",
        base="Does this review suggest the movie is appropriate for children?",
    ),
    Predicate(
        id="extra_rewatch",
        kind=KIND_YES_NO,
        population="general",
        base="Would the reviewer watch this movie again?",
    ),
    Predicate(
        id="extra_boring",
        kind=KIND_YES_NO,
        population="general",
        base="Does this review describe the movie as boring or dull?",
    ),
    Predicate(
        id="extra_plot",
        kind=KIND_YES_NO,
        population="general",
        base="Does this review praise the plot or the story?",
    ),
    Predicate(
        id="extra_disappointing",
        kind=KIND_YES_NO,
        population="general",
        base="Is the reviewer disappointed by the movie?",
    ),
    Predicate(
        id="extra_must_see",
        kind=KIND_YES_NO,
        population="general",
        base="Does this review present the movie as excellent or a must-see?",
    ),
)

PREDICATES_BY_ID: dict[str, Predicate] = {item.id: item for item in PREDICATES}


def select_predicates(ids: list[str] | None) -> list[Predicate]:
    """Return predicates in their canonical order. ``None`` selects all."""
    if not ids:
        return list(PREDICATES)
    unknown = sorted(set(ids) - PREDICATES_BY_ID.keys())
    if unknown:
        raise ValueError(f"Unknown predicate ids: {unknown}")
    wanted = set(ids)
    return [item for item in PREDICATES if item.id in wanted]

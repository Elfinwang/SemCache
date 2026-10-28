"""Base queries q over the TPC-H reviews (population ``tpch_reviews``)."""

from __future__ import annotations

from per_tuple_threshold.schema import KIND_YES_NO, Predicate

POPULATION = "tpch_reviews"

PREDICATES: tuple[Predicate, ...] = (
    Predicate(
        id="tpch_positive",
        kind=KIND_YES_NO,
        population=POPULATION,
        base="Determine if the following product review is clearly positive.",
    ),
)

PREDICATES_BY_ID: dict[str, Predicate] = {item.id: item for item in PREDICATES}


def select_predicates(ids: list[str] | None) -> list[Predicate]:
    """Return TPC-H predicates in their canonical order. ``None`` selects all."""
    if not ids:
        return list(PREDICATES)
    unknown = sorted(set(ids) - PREDICATES_BY_ID.keys())
    if unknown:
        raise ValueError(f"Unknown TPC-H predicate ids: {unknown}")
    wanted = set(ids)
    return [item for item in PREDICATES if item.id in wanted]

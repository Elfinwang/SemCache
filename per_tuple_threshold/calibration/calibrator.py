"""Fit t(d) for every calibration tuple and a pooled t(q) baseline.

For tuple d the fit uses the base query (s = 1, match by definition) and every
``fit`` variant: the observation is ``(s(q, q'), 1[LLM(q', d) == a_d])``.
The pooled baseline t(q) fits the same estimator on the union of those
observations over all tuples of the predicate, i.e. one global threshold.
"""

from __future__ import annotations

from typing import Iterable, Iterator, Mapping

from per_tuple_threshold.calibration.threshold import estimate_threshold
from per_tuple_threshold.llm.labeler import LabelJob, build_prompt
from per_tuple_threshold.schema import (
    CATEGORY_BASE,
    ROLE_FIT,
    DataTuple,
    Observation,
    Predicate,
    QueryVariant,
    ThresholdEstimate,
    TupleCalibration,
)

LabelKey = tuple[str, str, str]


def label_key(predicate: Predicate, tuple_id: str, question: str) -> LabelKey:
    return (predicate.kind, tuple_id, question)


def with_base_query(predicate: Predicate, variants: list[QueryVariant]) -> list[QueryVariant]:
    base = QueryVariant(
        name=CATEGORY_BASE,
        role=ROLE_FIT,
        category=CATEGORY_BASE,
        text=predicate.base,
        similarity=1.0,
    )
    return [base, *variants]


def label_jobs(
    predicate: Predicate,
    tuples: Iterable[DataTuple],
    variants: list[QueryVariant],
) -> Iterator[LabelJob]:
    queries = with_base_query(predicate, variants)
    for item in tuples:
        for query in queries:
            yield (
                label_key(predicate, item.tuple_id, query.text),
                build_prompt(predicate.kind, query.text, item.context),
                predicate.kind,
            )


def calibrate_tuple(
    predicate: Predicate,
    item: DataTuple,
    queries: list[QueryVariant],
    answers: Mapping[LabelKey, str],
) -> TupleCalibration:
    cached = answers[label_key(predicate, item.tuple_id, predicate.base)]
    observations = []
    for query in queries:
        answer = answers[label_key(predicate, item.tuple_id, query.text)]
        observations.append(Observation(
            name=query.name,
            role=query.role,
            category=query.category,
            question=query.text,
            similarity=query.similarity,
            target_similarity=query.target_similarity,
            answer=answer,
            match=answer == cached,
        ))
    fit = [obs for obs in observations if obs.role == ROLE_FIT]
    threshold = estimate_threshold([obs.similarity for obs in fit], [int(obs.match) for obs in fit])
    return TupleCalibration(
        tuple_id=item.tuple_id,
        summary=item.summary,
        text=item.text,
        cached_answer=cached,
        threshold=threshold,
        observations=tuple(observations),
    )


def calibrate_predicate(
    predicate: Predicate,
    tuples: Iterable[DataTuple],
    variants: list[QueryVariant],
    answers: Mapping[LabelKey, str],
) -> list[TupleCalibration]:
    queries = with_base_query(predicate, variants)
    return [calibrate_tuple(predicate, item, queries, answers) for item in tuples]


def pooled_threshold(records: Iterable[TupleCalibration]) -> ThresholdEstimate:
    similarities: list[float] = []
    labels: list[int] = []
    for record in records:
        for obs in record.fit_observations():
            similarities.append(obs.similarity)
            labels.append(int(obs.match))
    return estimate_threshold(similarities, labels)

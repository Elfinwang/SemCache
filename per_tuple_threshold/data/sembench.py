"""SemBench movie reviews and the frozen tuple sample used in the experiments.

``sample_ids.json`` maps a population name to an ordered list of tuple ids:

* ``general``: reviews of any movie (SemBench Q1, Q3, Q4, Q8).
* ``taken_3``: reviews of the movie ``taken_3`` (SemBench Q2).
* ``ant_man_pairs``: review pairs of ``ant_man_and_the_wasp_quantumania``
  (SemBench Q5, Q6, Q7). Pair ids have the form ``pair:<reviewId>:<reviewId>``.
* ``ant_man``: single reviews of the same movie (SemBench Q9, Q10).

Single-review ids have the form ``<movieId>:<reviewId>``. ``summary`` is the
SemBench ``scoreSentiment`` column and is informational only. The ``ant_man``
list contains two reviews twice; they are kept, so pooled fits count them twice.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from per_tuple_threshold.config import SAMPLE_IDS_PATH
from per_tuple_threshold.schema import DataTuple

ANT_MAN_MOVIE_ID = "ant_man_and_the_wasp_quantumania"
PAIR_PREFIX = "pair:"

ReviewIndex = dict[tuple[str, int], str]


def load_reviews(path: Path) -> ReviewIndex:
    """Index ``Reviews.csv`` by ``(movieId, reviewId)``. The first row wins."""
    frame = pd.read_csv(path)
    frame = frame.dropna(subset=["reviewText"])
    index: ReviewIndex = {}
    for row in frame.itertuples(index=False):
        index.setdefault((row.id, int(row.reviewId)), str(row.reviewText))
    return index


def load_populations(
    reviews: ReviewIndex,
    sample_path: Path = SAMPLE_IDS_PATH,
) -> dict[str, list[DataTuple]]:
    listed = json.loads(Path(sample_path).read_text())
    return {
        name: [build_tuple(item["tuple_id"], item["summary"], reviews) for item in rows]
        for name, rows in listed.items()
    }


def build_tuple(tuple_id: str, summary: str, reviews: ReviewIndex) -> DataTuple:
    if tuple_id.startswith(PAIR_PREFIX):
        _, left, right = tuple_id.split(":")
        first = _lookup(reviews, ANT_MAN_MOVIE_ID, int(left))
        second = _lookup(reviews, ANT_MAN_MOVIE_ID, int(right))
        return DataTuple(
            tuple_id=tuple_id,
            summary=summary,
            context=f"Review 1: {first}\nReview 2: {second}",
            text=f"{first} || {second}",
        )
    movie_id, review_id = tuple_id.rsplit(":", 1)
    text = _lookup(reviews, movie_id, int(review_id))
    return DataTuple(
        tuple_id=tuple_id,
        summary=summary,
        context=f"Review: {text}",
        text=text,
    )


def _lookup(reviews: ReviewIndex, movie_id: str, review_id: int) -> str:
    try:
        return reviews[(movie_id, review_id)]
    except KeyError as exc:
        raise KeyError(f"Review not found in Reviews.csv: {movie_id}:{review_id}") from exc

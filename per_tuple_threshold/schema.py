"""Domain objects shared by the calibration, prediction and serving stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

KIND_YES_NO = "yes_no"
KIND_SCORE = "score"

ROLE_FIT = "fit"
ROLE_EVAL = "eval"

CATEGORY_BASE = "base"


@dataclass(frozen=True)
class Predicate:
    """A base query q that the LLM evaluates on every tuple of one population."""

    id: str
    kind: str
    population: str
    base: str
    covers: tuple[str, ...] = ()


@dataclass(frozen=True)
class DataTuple:
    """One data item d. ``context`` is the exact text placed in the LLM prompt."""

    tuple_id: str
    summary: str
    context: str
    text: str


@dataclass(frozen=True)
class QueryVariant:
    """A query q' with its similarity s(q, q') to the base query.

    ``role`` is ``fit`` for queries used to estimate t(d) and ``eval`` for
    held-out queries that only score the reuse decision.
    """

    name: str
    role: str
    category: str
    text: str
    similarity: float
    target_similarity: float | None = None


@dataclass(frozen=True)
class ThresholdEstimate:
    """Output of the vCache estimator. ``t_hat is None`` means never reuse."""

    t_hat: float | None
    gamma: float | None
    var_t: float | None
    reason: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ThresholdEstimate":
        return cls(
            t_hat=payload["t_hat"],
            gamma=payload["gamma"],
            var_t=payload["var_t"],
            reason=payload["reason"],
        )


@dataclass(frozen=True)
class Observation:
    """LLM(q', d) for one variant and whether it equals the cached a_d."""

    name: str
    role: str
    category: str
    question: str
    similarity: float
    target_similarity: float | None
    answer: str
    match: bool

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Observation":
        return cls(**{name: payload[name] for name in cls.__dataclass_fields__})


@dataclass(frozen=True)
class TupleCalibration:
    tuple_id: str
    summary: str
    text: str
    cached_answer: str
    threshold: ThresholdEstimate
    observations: tuple[Observation, ...]

    def fit_observations(self) -> list[Observation]:
        return [item for item in self.observations if item.role == ROLE_FIT]

    def eval_observations(self) -> list[Observation]:
        return [item for item in self.observations if item.role == ROLE_EVAL]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TupleCalibration":
        return cls(
            tuple_id=payload["tuple_id"],
            summary=payload["summary"],
            text=payload["text"],
            cached_answer=payload["cached_answer"],
            threshold=ThresholdEstimate.from_dict(payload["threshold"]),
            observations=tuple(Observation.from_dict(item) for item in payload["observations"]),
        )


@dataclass
class PredicateCalibration:
    predicate: Predicate
    variants: list[QueryVariant]
    records: list[TupleCalibration]
    global_threshold: ThresholdEstimate
    held_out: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)

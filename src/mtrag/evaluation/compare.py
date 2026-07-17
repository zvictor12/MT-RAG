"""Paired comparisons for retrieval experiments."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import fmean

from .retrieval import RetrievalEvaluation


@dataclass(frozen=True)
class PairedBootstrapResult:
    metric: str
    cutoff: int
    query_count: int
    baseline_mean: float
    candidate_mean: float
    difference: float
    confidence: float
    confidence_low: float
    confidence_high: float
    probability_improvement: float
    samples: int
    seed: int


def _quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a quantile of an empty sample")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return (
        sorted_values[lower] * (1.0 - weight)
        + sorted_values[upper] * weight
    )


def paired_bootstrap(
    baseline: RetrievalEvaluation,
    candidate: RetrievalEvaluation,
    *,
    metric: str = "ndcg",
    cutoff: int = 5,
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> PairedBootstrapResult:
    """Compare two runs by resampling their aligned per-query differences."""
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")

    baseline_values = baseline.query_values(metric, cutoff)
    candidate_values = candidate.query_values(metric, cutoff)
    if baseline_values.keys() != candidate_values.keys():
        missing = sorted(baseline_values.keys() - candidate_values.keys())
        extra = sorted(candidate_values.keys() - baseline_values.keys())
        raise ValueError(
            "evaluations contain different qrels; "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    if not baseline_values:
        raise ValueError("evaluations contain no queries")

    keys = sorted(baseline_values)
    paired_values = [
        (baseline_values[key], candidate_values[key])
        for key in keys
    ]
    differences = [candidate - base for base, candidate in paired_values]
    rng = random.Random(seed)
    query_count = len(differences)
    bootstrap_differences = [
        fmean(differences[rng.randrange(query_count)] for _ in range(query_count))
        for _ in range(samples)
    ]
    bootstrap_differences.sort()

    alpha = (1.0 - confidence) / 2.0
    return PairedBootstrapResult(
        metric=metric.lower(),
        cutoff=cutoff,
        query_count=query_count,
        baseline_mean=fmean(base for base, _ in paired_values),
        candidate_mean=fmean(candidate for _, candidate in paired_values),
        difference=fmean(differences),
        confidence=confidence,
        confidence_low=_quantile(bootstrap_differences, alpha),
        confidence_high=_quantile(bootstrap_differences, 1.0 - alpha),
        probability_improvement=(
            sum(value > 0.0 for value in bootstrap_differences) / samples
        ),
        samples=samples,
        seed=seed,
    )

"""Adapter for the official IBM MT-RAG Task A evaluator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from mtrag.data.benchmark import COLLECTION_TO_DOMAIN

from .ibm import load_ibm_module


DEFAULT_CUTOFFS = (1, 3, 5, 10)


@dataclass(frozen=True)
class MetricValues:
    """Mean nDCG and recall values at fixed cutoffs."""

    ndcg: dict[int, float]
    recall: dict[int, float]


@dataclass(frozen=True)
class DomainEvaluation:
    domain: str
    query_count: int
    metrics: MetricValues


@dataclass(frozen=True)
class RetrievalEvaluation:
    cutoffs: tuple[int, ...]
    query_count: int
    metrics: MetricValues
    domains: dict[str, DomainEvaluation]


def _normalized_cutoffs(cutoffs: Sequence[int]) -> tuple[int, ...]:
    values = tuple(sorted(set(cutoffs)))
    if not values or any(value <= 0 for value in values):
        raise ValueError("cutoffs must contain positive integers")
    return values


def _mean_metrics(
    ndcg: Mapping[str, float],
    recall: Mapping[str, float],
    cutoffs: tuple[int, ...],
) -> MetricValues:
    return MetricValues(
        ndcg={cutoff: float(ndcg[f"NDCG@{cutoff}"]) for cutoff in cutoffs},
        recall={
            cutoff: float(recall[f"Recall@{cutoff}"])
            for cutoff in cutoffs
        },
    )


def evaluate_retrieval(
    benchmark_root: str | Path,
    prediction_path: str | Path,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
) -> RetrievalEvaluation:
    """Evaluate an official Task A prediction file with IBM's implementation."""
    root = Path(benchmark_root)
    prediction = Path(prediction_path)
    normalized_cutoffs = _normalized_cutoffs(cutoffs)
    official = load_ibm_module(root, "run_retrieval_eval.py")
    rankings, collections = official.prepare_results_dict(str(prediction))

    domains: dict[str, DomainEvaluation] = {}
    prediction_counts: dict[str, int] = {}
    for collection in sorted(set(collections.values())):
        try:
            domain = COLLECTION_TO_DOMAIN[collection]
        except KeyError as error:
            raise ValueError(f"unknown MT-RAG collection: {collection!r}") from error

        domain_rankings = {
            query_id: dict(rankings[query_id])
            for query_id, value in collections.items()
            if value == collection
        }
        qrels_path = (
            root
            / "mtrag-human"
            / "retrieval_tasks"
            / domain
            / "qrels"
            / "dev.tsv"
        )
        qrels = official.load_qrels(str(qrels_path))
        scores, ndcg, _, recall, _ = official.evaluate(
            qrels,
            domain_rankings,
            list(normalized_cutoffs),
        )

        domains[domain] = DomainEvaluation(
            domain=domain,
            query_count=len(scores),
            metrics=_mean_metrics(ndcg, recall, normalized_cutoffs),
        )
        prediction_counts[domain] = len(domain_rankings)

    total_predictions = sum(prediction_counts.values())
    if total_predictions == 0:
        raise ValueError(f"Task A prediction file is empty: {prediction}")

    metrics = MetricValues(
        ndcg={
            cutoff: sum(
                evaluation.metrics.ndcg[cutoff]
                * prediction_counts[domain]
                for domain, evaluation in domains.items()
            )
            / total_predictions
            for cutoff in normalized_cutoffs
        },
        recall={
            cutoff: sum(
                evaluation.metrics.recall[cutoff]
                * prediction_counts[domain]
                for domain, evaluation in domains.items()
            )
            / total_predictions
            for cutoff in normalized_cutoffs
        },
    )
    return RetrievalEvaluation(
        cutoffs=normalized_cutoffs,
        query_count=sum(domain.query_count for domain in domains.values()),
        metrics=metrics,
        domains=domains,
    )

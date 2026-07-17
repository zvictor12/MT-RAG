"""Exact, dependency-free Task A retrieval metrics."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import TypeAlias

from .qrels import Qrels, normalize_domain, normalize_task_id


DEFAULT_CUTOFFS = (1, 3, 5, 10)

ScoredRanking: TypeAlias = Mapping[str, float]
OrderedRanking: TypeAlias = Sequence[str]
Ranking: TypeAlias = ScoredRanking | OrderedRanking


@dataclass(frozen=True)
class MetricValues:
    """Mean or per-query nDCG and recall values at fixed cutoffs."""

    ndcg: dict[int, float]
    recall: dict[int, float]

    def value(self, metric: str, cutoff: int) -> float:
        values = {"ndcg": self.ndcg, "recall": self.recall}
        try:
            return values[metric.lower()][cutoff]
        except KeyError as error:
            raise ValueError(
                f"unknown metric/cutoff combination: {metric}@{cutoff}"
            ) from error


@dataclass(frozen=True)
class DomainEvaluation:
    domain: str
    query_count: int
    metrics: MetricValues
    per_query: dict[str, MetricValues]


@dataclass(frozen=True)
class RetrievalEvaluation:
    cutoffs: tuple[int, ...]
    query_count: int
    metrics: MetricValues
    domains: dict[str, DomainEvaluation]

    def query_values(self, metric: str, cutoff: int) -> dict[tuple[str, str], float]:
        """Return a stable key-to-score mapping for paired comparisons."""
        return {
            (domain, query_id): values.value(metric, cutoff)
            for domain, evaluation in self.domains.items()
            for query_id, values in evaluation.per_query.items()
        }


def _validate_cutoffs(cutoffs: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(sorted(set(cutoffs)))
    if not normalized or any(cutoff <= 0 for cutoff in normalized):
        raise ValueError("cutoffs must contain positive integers")
    return normalized


def rank_document_ids(ranking: Ranking | None) -> list[str]:
    """Produce a deterministic, duplicate-free document ranking."""
    if ranking is None:
        return []
    if isinstance(ranking, Mapping):
        scored: list[tuple[str, float]] = []
        for document_id, score in ranking.items():
            numeric_score = float(score)
            if not math.isfinite(numeric_score):
                raise ValueError(f"non-finite score for {document_id!r}")
            scored.append((str(document_id), numeric_score))
        return [
            document_id
            for document_id, _ in sorted(
                scored,
                key=lambda item: (-item[1], item[0]),
            )
        ]

    if isinstance(ranking, (str, bytes)):
        raise TypeError("an ordered ranking must be a sequence of document IDs")

    result: list[str] = []
    seen: set[str] = set()
    for document_id in ranking:
        normalized = str(document_id)
        if normalized not in seen:
            result.append(normalized)
            seen.add(normalized)
    return result


def _dcg(relevances: Sequence[int]) -> float:
    return sum(
        (2**relevance - 1) / math.log2(rank + 1)
        for rank, relevance in enumerate(relevances, start=1)
        if relevance > 0
    )


def score_query(
    relevance: Mapping[str, int],
    ranking: Ranking | None,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
) -> MetricValues:
    """Compute nDCG and recall for one query."""
    normalized_cutoffs = _validate_cutoffs(cutoffs)
    documents = rank_document_ids(ranking)
    relevant_documents = {
        document_id
        for document_id, score in relevance.items()
        if score > 0
    }
    ideal_relevances = sorted(
        (score for score in relevance.values() if score > 0),
        reverse=True,
    )

    ndcg: dict[int, float] = {}
    recall: dict[int, float] = {}
    for cutoff in normalized_cutoffs:
        predicted_relevances = [
            relevance.get(document_id, 0)
            for document_id in documents[:cutoff]
        ]
        ideal = _dcg(ideal_relevances[:cutoff])
        ndcg[cutoff] = _dcg(predicted_relevances) / ideal if ideal else 0.0

        retrieved_relevant = len(
            relevant_documents.intersection(documents[:cutoff])
        )
        recall[cutoff] = (
            retrieved_relevant / len(relevant_documents)
            if relevant_documents
            else 0.0
        )
    return MetricValues(ndcg=ndcg, recall=recall)


def _mean_metrics(
    values: Sequence[MetricValues],
    cutoffs: tuple[int, ...],
) -> MetricValues:
    if not values:
        return MetricValues(
            ndcg={cutoff: 0.0 for cutoff in cutoffs},
            recall={cutoff: 0.0 for cutoff in cutoffs},
        )
    return MetricValues(
        ndcg={
            cutoff: fmean(value.ndcg[cutoff] for value in values)
            for cutoff in cutoffs
        },
        recall={
            cutoff: fmean(value.recall[cutoff] for value in values)
            for cutoff in cutoffs
        },
    )


def evaluate_retrieval(
    qrels_by_domain: Mapping[str, Qrels],
    rankings_by_domain: Mapping[str, Mapping[str, Ranking]],
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
) -> RetrievalEvaluation:
    """Evaluate all qrels, assigning zero to every missing prediction."""
    normalized_cutoffs = _validate_cutoffs(cutoffs)
    normalized_rankings: dict[str, dict[str, Ranking]] = {}
    for domain, rankings in rankings_by_domain.items():
        canonical_domain = normalize_domain(domain)
        domain_rankings = normalized_rankings.setdefault(canonical_domain, {})
        for query_id, ranking in rankings.items():
            canonical_query_id = normalize_task_id(query_id)
            if canonical_query_id in domain_rankings:
                raise ValueError(
                    f"duplicate prediction after ID normalization: "
                    f"{canonical_query_id}"
                )
            domain_rankings[canonical_query_id] = ranking

    domains: dict[str, DomainEvaluation] = {}
    all_query_metrics: list[MetricValues] = []
    for domain, qrels in qrels_by_domain.items():
        canonical_domain = normalize_domain(domain)
        predictions = normalized_rankings.get(canonical_domain, {})
        per_query: dict[str, MetricValues] = {}
        for query_id, relevance in qrels.items():
            canonical_query_id = normalize_task_id(query_id)
            per_query[canonical_query_id] = score_query(
                relevance,
                predictions.get(canonical_query_id),
                normalized_cutoffs,
            )

        query_metrics = list(per_query.values())
        all_query_metrics.extend(query_metrics)
        domains[canonical_domain] = DomainEvaluation(
            domain=canonical_domain,
            query_count=len(per_query),
            metrics=_mean_metrics(query_metrics, normalized_cutoffs),
            per_query=per_query,
        )

    return RetrievalEvaluation(
        cutoffs=normalized_cutoffs,
        query_count=len(all_query_metrics),
        metrics=_mean_metrics(all_query_metrics, normalized_cutoffs),
        domains=domains,
    )


def load_rankings_jsonl(path: str | Path) -> dict[str, dict[str, ScoredRanking]]:
    """Load an official Task A prediction JSONL into scored rankings."""
    rankings: dict[str, dict[str, ScoredRanking]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            query_id = normalize_task_id(item["task_id"])
            domain = normalize_domain(item["Collection"])

            document_scores: dict[str, float] = {}
            for context in item.get("contexts", []):
                document_scores[str(context["document_id"])] = float(
                    context["score"]
                )

            domain_rankings = rankings.setdefault(domain, {})
            if query_id in domain_rankings:
                raise ValueError(
                    f"duplicate task_id at {path}:{line_number}: {query_id}"
                )
            domain_rankings[query_id] = document_scores
    return rankings

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mtrag.config import settings
from mtrag.data.jsonl import read_jsonl, write_jsonl
from mtrag.evaluation import (
    DEFAULT_CUTOFFS,
    COLLECTION_TO_DOMAIN,
    MetricValues,
    RetrievalEvaluation,
    evaluate_retrieval,
    load_benchmark_qrels,
    load_rankings_jsonl,
    normalize_domain,
    normalize_task_id,
)


def metric_values_dict(values: MetricValues) -> dict[str, dict[str, float]]:
    return {
        "ndcg": {str(cutoff): score for cutoff, score in values.ndcg.items()},
        "recall": {
            str(cutoff): score for cutoff, score in values.recall.items()
        },
    }


def evaluation_dict(report: RetrievalEvaluation) -> dict[str, Any]:
    return {
        "cutoffs": list(report.cutoffs),
        "query_count": report.query_count,
        "metrics": metric_values_dict(report.metrics),
        "domains": {
            domain: {
                "query_count": domain_report.query_count,
                "metrics": metric_values_dict(domain_report.metrics),
                "per_query": {
                    query_id: metric_values_dict(values)
                    for query_id, values in domain_report.per_query.items()
                },
            }
            for domain, domain_report in report.domains.items()
        },
    }


def official_query_scores(values: MetricValues) -> dict[str, float]:
    scores: dict[str, float] = {}
    for cutoff, value in values.ndcg.items():
        scores[f"ndcg_cut_{cutoff}"] = value
    for cutoff, value in values.recall.items():
        scores[f"recall_{cutoff}"] = value
    return scores


def write_enriched_predictions(
    input_path: Path,
    output_path: Path,
    report: RetrievalEvaluation,
) -> None:
    rows = read_jsonl(input_path)
    for row in rows:
        domain = normalize_domain(row["Collection"])
        query_id = normalize_task_id(row["task_id"])
        values = report.domains[domain].per_query.get(query_id)
        row["retriever_scores"] = (
            official_query_scores(values) if values is not None else {}
        )
    write_jsonl(output_path, rows)


def write_aggregate_csv(
    path: Path,
    report: RetrievalEvaluation,
) -> None:
    domain_to_collection = {
        domain: collection
        for collection, domain in COLLECTION_TO_DOMAIN.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("nDCG", "Recall", "collection", "count"),
        )
        writer.writeheader()
        for domain, domain_report in report.domains.items():
            writer.writerow(
                {
                    "nDCG": [
                        domain_report.metrics.ndcg[k] for k in report.cutoffs
                    ],
                    "Recall": [
                        domain_report.metrics.recall[k] for k in report.cutoffs
                    ],
                    "collection": domain_to_collection[domain],
                    "count": domain_report.query_count,
                }
            )
        writer.writerow(
            {
                "nDCG": [report.metrics.ndcg[k] for k in report.cutoffs],
                "Recall": [report.metrics.recall[k] for k in report.cutoffs],
                "collection": "all",
                "count": report.query_count,
            }
        )


def print_report(report: RetrievalEvaluation) -> None:
    headings = ["scope", "queries"] + [
        f"{metric}@{cutoff}"
        for cutoff in report.cutoffs
        for metric in ("nDCG", "Recall")
    ]
    print("\t".join(headings))
    rows: list[tuple[str, int, MetricValues]] = [
        (domain, values.query_count, values.metrics)
        for domain, values in report.domains.items()
    ]
    rows.append(("all", report.query_count, report.metrics))
    for scope, count, metrics in rows:
        values: list[str] = [scope, str(count)]
        for cutoff in report.cutoffs:
            values.extend(
                (
                    f"{metrics.ndcg[cutoff]:.6f}",
                    f"{metrics.recall[cutoff]:.6f}",
                )
            )
        print("\t".join(values))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MT-RAG Task A retrieval predictions.",
    )
    parser.add_argument("-i", "--input", type=Path, required=True)
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=settings.benchmark_root,
    )
    parser.add_argument(
        "--cutoffs",
        type=int,
        nargs="+",
        default=list(DEFAULT_CUTOFFS),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional machine-readable JSON report.",
    )
    parser.add_argument(
        "--official-output",
        type=Path,
        help=(
            "Optional IBM-compatible JSONL with per-task retriever_scores. "
            "An aggregate CSV is written beside it."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    qrels = load_benchmark_qrels(args.benchmark_root)
    rankings = load_rankings_jsonl(args.input)
    report = evaluate_retrieval(qrels, rankings, args.cutoffs)
    print_report(report)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(evaluation_dict(report), ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )

    if args.official_output is not None:
        write_enriched_predictions(args.input, args.official_output, report)
        aggregate_path = args.official_output.with_name(
            f"{args.official_output.stem}_aggregate.csv"
        )
        write_aggregate_csv(aggregate_path, report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from mtrag.config import settings
from mtrag.evaluation import DEFAULT_CUTOFFS, RetrievalEvaluation, evaluate_retrieval
from mtrag.runtime.state import write_json_atomic


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Task A with IBM's MT-RAG implementation."
    )
    parser.add_argument("-i", "--input", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path)
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
    return parser.parse_args(argv)


def print_report(report: RetrievalEvaluation) -> None:
    headers = ["scope", "queries"] + [
        f"{metric}@{cutoff}"
        for cutoff in report.cutoffs
        for metric in ("nDCG", "Recall")
    ]
    print("\t".join(headers))
    rows = [
        (domain, values.query_count, values.metrics)
        for domain, values in report.domains.items()
    ] + [("all", report.query_count, report.metrics)]
    for scope, count, metrics in rows:
        values = [scope, str(count)]
        for cutoff in report.cutoffs:
            values.extend(
                (f"{metrics.ndcg[cutoff]:.6f}", f"{metrics.recall[cutoff]:.6f}")
            )
        print("\t".join(values))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = evaluate_retrieval(
        args.benchmark_root,
        args.input,
        args.cutoffs,
    )
    print_report(report)
    if args.output is not None:
        write_json_atomic(args.output, asdict(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

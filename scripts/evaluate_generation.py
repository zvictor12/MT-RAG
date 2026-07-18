from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Sequence
from pathlib import Path

from mtrag.data.jsonl import read_jsonl, write_jsonl
from mtrag.evaluation import (
    AlgorithmicGenerationEvaluator,
    BertScoreBatcher,
    summarize_generation_metrics,
)
from mtrag.runtime import ThermalGuard
from mtrag.runtime.state import write_json_atomic


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the algorithmic MT-RAG Task B/C metrics with batched "
            "BERTScore inference."
        ),
    )
    parser.add_argument("-i", "--input", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument(
        "--summary",
        type=Path,
        help="Optional aggregate JSON path; defaults beside --output.",
    )
    parser.add_argument(
        "--model",
        default="microsoft/deberta-xlarge-mnli",
    )
    parser.add_argument(
        "--device",
        default=os.getenv("BERTSCORE_DEVICE", "cuda:0"),
    )
    parser.add_argument("--batch-size", type=positive_int, default=4)
    parser.add_argument(
        "--chunk-size",
        type=positive_int,
        help="Pairs between thermal checks; defaults to 8 model batches.",
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        help="Evaluate only the first N records for a smoke test.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    records = read_jsonl(args.input)
    if args.limit is not None:
        records = records[: args.limit]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.info(
        "evaluating %d records with batched BERTScore on %s",
        len(records),
        args.device,
    )
    semantic_scorer = BertScoreBatcher(
        model_type=args.model,
        device=args.device,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        guard=ThermalGuard(),
    )
    evaluated = AlgorithmicGenerationEvaluator(semantic_scorer).evaluate(records)
    write_jsonl(args.output, evaluated)
    summary_path = args.summary or args.output.with_name(
        f"{args.output.stem}_summary.json"
    )
    write_json_atomic(summary_path, summarize_generation_metrics(evaluated))
    logging.info("wrote %d evaluated records to %s", len(evaluated), args.output)
    logging.info("wrote aggregate task means to %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

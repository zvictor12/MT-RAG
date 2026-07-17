from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


RETRIEVAL_CUTOFFS = (1, 3, 5, 10)
GENERATION_METRICS = (
    "Recall",
    "RougeL_stemFalse",
    "BertscoreP",
    "BertscoreR",
    "BertKPrec",
    "Extractiveness_RougeL",
    "RB_agg",
    "Length",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(
    metrics: Mapping[str, Any],
    name: str,
    cutoff: int | None = None,
) -> float | None:
    value = metrics.get(name)
    if cutoff is not None and isinstance(value, Mapping):
        value = value.get(str(cutoff))
    elif isinstance(value, Mapping):
        value = value.get("mean")
    return float(value) if isinstance(value, (int, float)) else None


def _number(value: float | None, *, length: bool = False) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}" if length else f"{value:.4f}"


def _table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    materialized = [tuple(str(value) for value in row) for row in rows]
    widths = [len(header) for header in headers]
    for row in materialized:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def render(row: Sequence[str]) -> str:
        return "  ".join(
            value.ljust(widths[index])
            for index, value in enumerate(row)
        ).rstrip()

    divider = tuple("-" * width for width in widths)
    return "\n".join((render(headers), render(divider), *(render(row) for row in materialized)))


def _retrieval_report(run_dir: Path) -> str | None:
    paths = sorted((run_dir / "evaluation" / "retrieval").glob("*.json"))
    if not paths:
        return None

    headers = (
        "variant",
        *(f"nDCG@{cutoff}" for cutoff in RETRIEVAL_CUTOFFS),
        *(f"R@{cutoff}" for cutoff in RETRIEVAL_CUTOFFS),
    )
    rows = []
    query_counts: set[int] = set()
    for path in paths:
        report = _read_json(path)
        query_count = report.get("query_count")
        if isinstance(query_count, int):
            query_counts.add(query_count)
        metrics = report.get("metrics") or {}
        rows.append(
            (
                path.stem,
                *(
                    _number(_metric(metrics, "ndcg", cutoff))
                    for cutoff in RETRIEVAL_CUTOFFS
                ),
                *(
                    _number(_metric(metrics, "recall", cutoff))
                    for cutoff in RETRIEVAL_CUTOFFS
                ),
            )
        )
    count = str(next(iter(query_counts))) if len(query_counts) == 1 else "mixed"
    return f"RETRIEVAL (queries: {count})\n{_table(headers, rows)}"


def _generation_report(run_dir: Path) -> str | None:
    paths = sorted((run_dir / "evaluation" / "generation").glob("*.json"))
    if not paths:
        return None

    labels = {
        "RougeL_stemFalse": "RougeL",
        "Extractiveness_RougeL": "Extract.",
        "Length": "Length",
    }
    headers = (
        "task",
        "count",
        *(labels.get(name, name) for name in GENERATION_METRICS),
    )
    rows = []
    for path in paths:
        report = _read_json(path)
        metrics = report.get("metrics") or {}
        rows.append(
            (
                path.stem,
                str(report.get("task_count", "-")),
                *(
                    _number(_metric(metrics, name), length=name == "Length")
                    for name in GENERATION_METRICS
                ),
            )
        )
    return f"GENERATION\n{_table(headers, rows)}"


def _decisions_report(run_dir: Path) -> str | None:
    directory = run_dir / "decisions"
    lines: list[str] = []

    winner_path = directory / "bge-winner.json"
    if winner_path.exists():
        winner = _read_json(winner_path)
        lines.append(
            "BGE winner: "
            f"{winner.get('winner', '-')} "
            f"({winner.get('metric', 'metric')}={_number(winner.get('score'))})"
        )

    rewrite_path = directory / "rewrite-winner.json"
    if rewrite_path.exists():
        rewrite = _read_json(rewrite_path)
        lines.append(
            "Rewrite winner: "
            f"{rewrite.get('winner', '-')} "
            f"({rewrite.get('metric', 'metric')}={_number(rewrite.get('score'))})"
        )

    reranker_variants = directory / "reranker-variants.json"
    reranker_path = directory / "reranker.json"
    if reranker_variants.exists():
        reranker = _read_json(reranker_variants)
        for variant, decision in reranker.get("variants", {}).items():
            lines.append(
                f"Reranker {variant}: "
                f"{'enabled' if decision.get('enabled') else 'disabled'}, "
                f"nDCG@5 gain={_number(decision.get('ndcg5_gain'))}, "
                "P(improvement)="
                f"{_number(decision.get('probability_improvement'))}"
            )
    elif reranker_path.exists():
        reranker = _read_json(reranker_path)
        lines.append(
            "Reranker: "
            f"{'enabled' if reranker.get('enabled') else 'disabled'}, "
            f"nDCG@5 gain={_number(reranker.get('ndcg5_gain'))}, "
            "P(improvement)="
            f"{_number(reranker.get('probability_improvement'))}"
        )

    final_path = directory / "winner.json"
    if final_path.exists():
        final = _read_json(final_path)
        score = (final.get("scores") or {}).get(final.get("winner"))
        lines.append(
            f"Final winner: {final.get('winner', '-')} "
            f"({final.get('metric', 'metric')}={_number(score)})"
        )

    return "DECISIONS\n" + "\n".join(lines) if lines else None


def render_experiment_results(run_dir: Path) -> str:
    sections = [
        section
        for section in (
            _retrieval_report(run_dir),
            _generation_report(run_dir),
            _decisions_report(run_dir),
        )
        if section
    ]
    if not sections:
        raise FileNotFoundError(f"no experiment results found in {run_dir}")
    return "\n\n".join(sections)

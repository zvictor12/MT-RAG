"""Write predictions accepted by the official MT-RAG format checker."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from mtrag.data.jsonl import write_jsonl


MAX_OFFICIAL_CONTEXTS = 10


def deterministic_rank_score(rank: int) -> float:
    """Return a unique score that preserves an already-decided rank."""
    return 1.0 / rank


def make_retrieval_record(
    base_record: Mapping[str, Any],
    contexts: Sequence[Mapping[str, Any]],
    *,
    max_contexts: int = MAX_OFFICIAL_CONTEXTS,
) -> dict[str, Any]:
    """Attach ranked contexts without mutating the benchmark input record.

    Raw retriever scores are deliberately replaced with reciprocal rank scores.
    This prevents score ties or score scales from changing the intended order in
    the official evaluator, which reconstructs rankings from the score field.
    """
    official_contexts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for context in contexts:
        document_id = context["document_id"]
        if document_id in seen:
            continue

        output_context = dict(context)
        output_context["score"] = deterministic_rank_score(
            len(official_contexts) + 1
        )
        official_contexts.append(output_context)
        seen.add(document_id)
        if len(official_contexts) == max_contexts:
            break

    record = dict(base_record)
    record.pop("targets", None)
    record.pop("enrichments", None)
    record.pop("predictions", None)
    record.pop("retriever_scores", None)
    record["contexts"] = official_contexts
    return record


def write_retrieval_jsonl(
    path: str | Path,
    records: Iterable[Mapping[str, Any]],
) -> int:
    """Write official Task A records atomically and return their count."""
    rows = list(records)
    write_jsonl(path, rows)
    return len(rows)

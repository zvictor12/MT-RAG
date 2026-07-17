"""Write predictions accepted by the official MT-RAG format checker."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


MAX_OFFICIAL_CONTEXTS = 10


def deterministic_rank_score(rank: int) -> float:
    """Return a unique score that preserves an already-decided rank."""
    if rank <= 0:
        raise ValueError("rank must be positive")
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
    if max_contexts <= 0 or max_contexts > MAX_OFFICIAL_CONTEXTS:
        raise ValueError(
            f"max_contexts must be between 1 and {MAX_OFFICIAL_CONTEXTS}"
        )
    for field in ("task_id", "Collection"):
        if not isinstance(base_record.get(field), str):
            raise ValueError(f"base record requires string field {field!r}")

    official_contexts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for context in contexts:
        document_id = context.get("document_id")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError("each context requires a non-empty document_id")
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
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")

    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            contexts = record.get("contexts")
            if not isinstance(record.get("task_id"), str):
                raise ValueError("record requires a string task_id")
            if not isinstance(record.get("Collection"), str):
                raise ValueError("record requires a string Collection")
            if not isinstance(contexts, list):
                raise ValueError("record requires a contexts list")
            if len(contexts) > MAX_OFFICIAL_CONTEXTS:
                raise ValueError(
                    f"official Task A allows at most {MAX_OFFICIAL_CONTEXTS} contexts"
                )
            for context in contexts:
                if not isinstance(context.get("document_id"), str):
                    raise ValueError("context requires a string document_id")
                score = context.get("score")
                if not isinstance(score, (int, float)) or not math.isfinite(score):
                    raise ValueError("context requires a finite numeric score")

            handle.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            )
            handle.write("\n")
            count += 1

    temporary.replace(destination)
    return count

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from mtrag.data.jsonl import read_jsonl
from mtrag.evaluation import make_retrieval_record, write_retrieval_jsonl
from mtrag.schemas import BenchmarkTask, Context, SearchHit


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    """Content-addressed paths inside one experiment campaign."""

    root: Path

    @property
    def cache(self) -> Path:
        return self.root / "cache.sqlite"

    def rewrite(self, name: str, revision: str) -> Path:
        return self.root / "rewrites" / name / revision / "queries.jsonl"

    def bge_features(self, name: str, revision: str) -> Path:
        return self.root / "features" / "bge" / name / revision

    def experiment(self, reference: str, revision: str) -> Path:
        pipeline_and_output = reference.replace(".", "/", 1)
        return self.root / "experiments" / pipeline_and_output / revision

    def candidates(self, reference: str, revision: str) -> Path:
        return self.experiment(reference, revision) / "candidates.jsonl"

    def prediction(self, reference: str, revision: str) -> Path:
        return self.experiment(reference, revision) / "task-a.jsonl"

    def retrieval_report(
        self,
        reference: str,
        output_revision: str,
        evaluation_revision: str,
    ) -> Path:
        return (
            self.experiment(reference, output_revision)
            / "evaluation"
            / evaluation_revision
            / "task-a-metrics.json"
        )

    def generation(self, job: str, revision: str) -> Path:
        return self.root / "generation" / job / revision / "predictions.jsonl"

    def generation_metrics(
        self,
        job: str,
        generation_revision: str,
        evaluation_revision: str,
    ) -> Path:
        return (
            self.root
            / "generation"
            / job
            / generation_revision
            / "evaluation"
            / evaluation_revision
            / "ibm-metrics.jsonl"
        )

    def generation_summary(
        self,
        job: str,
        generation_revision: str,
        evaluation_revision: str,
    ) -> Path:
        return (
            self.root
            / "generation"
            / job
            / generation_revision
            / "evaluation"
            / evaluation_revision
            / "ibm-summary.json"
        )

    def stage_marker(self, fingerprint: str) -> Path:
        return self.root / "state" / "completed" / f"{fingerprint}.json"

    def create_directories(self) -> None:
        (self.root / "state" / "completed").mkdir(parents=True, exist_ok=True)


class JsonlCheckpoint:
    """Append-only JSONL output that repairs an interrupted final write."""

    def __init__(self, path: str | Path, *, key: str = "task_id") -> None:
        self.path = Path(path)
        self.key = key
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records = self._load_and_repair()

    @property
    def completed(self) -> set[str]:
        return set(self.records)

    def append(self, record: Mapping[str, Any]) -> None:
        self.append_many((record,))

    def append_many(self, records: Sequence[Mapping[str, Any]]) -> None:
        pending = [(self._key(record), dict(record)) for record in records]
        if not pending:
            return
        keys = [key for key, _record in pending]
        if len(keys) != len(set(keys)) or any(key in self.records for key in keys):
            raise ValueError("duplicate checkpoint key")

        payload = "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for _key, record in pending
        ).encode("utf-8")
        with self.path.open("ab") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        self.records.update(pending)

    def _key(self, record: Mapping[str, Any]) -> str:
        value = record.get(self.key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"checkpoint record requires string field {self.key!r}")
        return value

    def _load_and_repair(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}

        data = self.path.read_bytes()
        if data and not data.endswith(b"\n"):
            prefix, separator, tail = data.rpartition(b"\n")
            try:
                json.loads(tail)
            except (json.JSONDecodeError, UnicodeDecodeError):
                self.path.write_bytes(prefix + separator)
            else:
                with self.path.open("ab") as handle:
                    handle.write(b"\n")
                    handle.flush()
                    os.fsync(handle.fileno())

        rows = read_jsonl(self.path)
        records = {self._key(record): record for record in rows}
        if len(records) != len(rows):
            raise ValueError(f"duplicate checkpoint key: {self.path}")
        return records


def task_record(task: BenchmarkTask, *, include_targets: bool = False) -> dict[str, Any]:
    record: dict[str, Any] = {
        "conversation_id": task.conversation_id,
        "task_id": task.task_id,
        "turn": task.turn,
        "Collection": task.collection,
        "input": [
            {"speaker": message.speaker, "text": message.text}
            for message in task.messages
        ],
    }
    if include_targets:
        record["targets"] = [
            {"speaker": message.speaker, "text": message.text}
            for message in task.targets
        ]
        record["Answerability"] = list(task.answerability)
    return record


def hit_context(hit: SearchHit) -> dict[str, Any]:
    context: dict[str, Any] = {
        "document_id": hit.document_id,
        "score": 1.0 / hit.rank,
        "retriever_score": hit.score,
        "rank": hit.rank,
        "source": hit.source,
    }
    if hit.title is not None:
        context["title"] = hit.title
    if hit.text is not None:
        context["text"] = hit.text
    if hit.components:
        context["components"] = hit.components
    return context


def ranking_record(task: BenchmarkTask, hits: Sequence[SearchHit]) -> dict[str, Any]:
    record = task_record(task)
    record["contexts"] = [hit_context(hit) for hit in hits]
    return record


def record_hits(record: Mapping[str, Any]) -> list[SearchHit]:
    return [
        SearchHit(
            document_id=str(context["document_id"]),
            score=float(context.get("retriever_score", context.get("score", 0.0))),
            rank=rank,
            source=str(context.get("source", "artifact")),
            title=context.get("title"),
            text=context.get("text"),
            components={
                str(key): float(value)
                for key, value in context.get("components", {}).items()
            },
        )
        for rank, context in enumerate(record.get("contexts", ()), start=1)
    ]


def context_from_hit(hit: SearchHit) -> Context:
    return Context(
        document_id=hit.document_id,
        text=hit.text or "",
        title=hit.title,
        score=hit.score,
    )


def context_record(context: Context, rank: int) -> dict[str, Any]:
    record: dict[str, Any] = {
        "document_id": context.document_id,
        "text": context.text,
        "score": 1.0 / rank,
    }
    if context.score is not None:
        record["retriever_score"] = context.score
    if context.title is not None:
        record["title"] = context.title
    if context.url is not None:
        record["url"] = context.url
    return record


def materialize_prediction(
    candidate_path: Path,
    prediction_path: Path,
    *,
    top_k: int,
    tasks: Sequence[BenchmarkTask] | None = None,
) -> int:
    candidates = read_jsonl(candidate_path)
    if tasks is None:
        source_records = candidates
    else:
        by_id = {record["task_id"]: record for record in candidates}
        if len(by_id) != len(candidates):
            raise ValueError(f"duplicate task_id in {candidate_path}")
        task_ids = {task.task_id for task in tasks}
        unknown = sorted(by_id.keys() - task_ids)
        if unknown:
            raise ValueError(
                f"unknown task_id in {candidate_path}: {unknown[0]}"
            )
        source_records = [
            by_id.get(task.task_id, task_record(task))
            for task in tasks
        ]
    records = (
        make_retrieval_record(
            record,
            [
                {"document_id": context["document_id"]}
                for context in record.get("contexts", ())
            ],
            max_contexts=top_k,
        )
        for record in source_records
    )
    return write_retrieval_jsonl(prediction_path, records)

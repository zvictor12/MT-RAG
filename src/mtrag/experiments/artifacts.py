from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mtrag.evaluation import make_retrieval_record, write_retrieval_jsonl
from mtrag.schemas import BenchmarkTask, Context, SearchHit


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    """Paths for independently versioned experiment artifacts.

    A campaign may contain many query, retrieval and generation experiments.
    The semantic fingerprint addresses one immutable revision; changing an
    unrelated experiment never invalidates the rest of the campaign.
    """

    root: Path

    @property
    def cache(self) -> Path:
        return self.root / "cache.sqlite"

    def rewrite(self, name: str, revision: str) -> Path:
        return self._revision("rewrites", name, revision) / "queries.jsonl"

    def bge_features(self, name: str, revision: str) -> Path:
        return self._revision("features/bge", name, revision)

    def experiment(self, reference: str, revision: str) -> Path:
        pipeline, output = _retrieval_reference(reference)
        return self._revision(f"experiments/{pipeline}", output, revision)

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

    def generation_dir(self, job: str, revision: str) -> Path:
        return self._revision("generation", job, revision)

    def generation(self, job: str, revision: str) -> Path:
        return self.generation_dir(job, revision) / "predictions.jsonl"

    def generation_metrics(
        self,
        job: str,
        generation_revision: str,
        evaluation_revision: str,
    ) -> Path:
        return (
            self.generation_dir(job, generation_revision)
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
            self.generation_dir(job, generation_revision)
            / "evaluation"
            / evaluation_revision
            / "ibm-summary.json"
        )

    def stage_marker(self, fingerprint: str) -> Path:
        _fingerprint(fingerprint)
        return self.root / "state" / "completed" / f"{fingerprint}.json"

    def _revision(self, category: str, name: str, revision: str) -> Path:
        _safe_parts(category)
        _safe_parts(name)
        _fingerprint(revision)
        return self.root / category / name / revision

    def create_directories(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "state" / "completed").mkdir(parents=True, exist_ok=True)


def _safe_parts(value: str) -> None:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"unsafe artifact name: {value!r}")
    parts = path.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"unsafe artifact name: {value!r}")


def _fingerprint(value: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"invalid artifact fingerprint: {value!r}")


def _retrieval_reference(reference: str) -> tuple[str, str]:
    pipeline, separator, output = reference.partition(".")
    if not separator or "." in output:
        raise ValueError(f"invalid retrieval reference: {reference!r}")
    _safe_parts(pipeline)
    _safe_parts(output)
    return pipeline, output


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
        pending: list[tuple[str, dict[str, Any]]] = []
        batch_keys: set[str] = set()
        for record in records:
            raw_key = record.get(self.key)
            if not isinstance(raw_key, str) or not raw_key:
                raise ValueError(
                    f"checkpoint record requires string field {self.key!r}"
                )
            if raw_key in self.records or raw_key in batch_keys:
                raise ValueError(f"duplicate checkpoint key: {raw_key}")
            batch_keys.add(raw_key)
            pending.append((raw_key, dict(record)))
        if not pending:
            return

        payload = "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for _key, record in pending
        ).encode("utf-8")
        with self.path.open("ab") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        self.records.update(pending)

    def _load_and_repair(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}

        data = self.path.read_bytes()
        records: dict[str, dict[str, Any]] = {}
        valid_bytes = 0
        lines = data.splitlines(keepends=True)
        for index, raw_line in enumerate(lines):
            if not raw_line.strip():
                valid_bytes += len(raw_line)
                continue
            try:
                record = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if any(line.strip() for line in lines[index + 1 :]):
                    raise ValueError(f"corrupt JSONL checkpoint: {self.path}") from None
                break
            raw_key = record.get(self.key)
            if not isinstance(raw_key, str) or not raw_key:
                raise ValueError(
                    f"checkpoint record requires string field {self.key!r}: {self.path}"
                )
            if raw_key in records:
                raise ValueError(f"duplicate checkpoint key {raw_key!r}: {self.path}")
            records[raw_key] = record
            valid_bytes += len(raw_line)

        if valid_bytes != len(data):
            with self.path.open("r+b") as handle:
                handle.truncate(valid_bytes)
        elif data and not data.endswith(b"\n"):
            with self.path.open("ab") as handle:
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
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


def hit_context(hit: SearchHit, *, score: float | None = None) -> dict[str, Any]:
    context: dict[str, Any] = {
        "document_id": hit.document_id,
        "score": 1.0 / hit.rank if score is None else score,
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
    hits = []
    for rank, context in enumerate(record.get("contexts", ()), start=1):
        hits.append(
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
        )
    return hits


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


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
    return records


def write_jsonl_atomic(path: str | Path, records: Iterable[Mapping[str, Any]]) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)
    return count

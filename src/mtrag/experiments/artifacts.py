from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from mtrag.data.jsonl import read_jsonl
from mtrag.evaluation import make_retrieval_record, write_retrieval_jsonl
from mtrag.schemas import (
    ArtifactRef,
    BenchmarkTask,
    Context,
    RankedCandidates,
    SearchHit,
)


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    """All content-addressed paths inside one experiment campaign."""

    root: Path

    @property
    def cache(self) -> Path:
        return self.root / "cache.sqlite"

    def rewrite(self, artifact: ArtifactRef) -> Path:
        return (
            self.root
            / "rewrites"
            / artifact.name
            / artifact.revision
            / "queries.jsonl"
        )

    def bge_features(self, artifact: ArtifactRef) -> Path:
        return self.root / "features" / "bge" / artifact.name / artifact.revision

    def experiment(self, artifact: ArtifactRef) -> Path:
        pipeline, output = artifact.name.split(".", 1)
        return self.root / "experiments" / pipeline / output / artifact.revision

    def candidates(self, artifact: ArtifactRef) -> Path:
        return self.experiment(artifact) / "candidates.jsonl"

    def prediction(self, artifact: ArtifactRef) -> Path:
        return self.experiment(artifact) / "task-a.jsonl"

    def retrieval_report(self, artifact: ArtifactRef, evaluation: str) -> Path:
        return (
            self.experiment(artifact)
            / "evaluation"
            / evaluation
            / "task-a-metrics.json"
        )

    def generation(self, artifact: ArtifactRef) -> Path:
        return (
            self.root
            / "generation"
            / artifact.name
            / artifact.revision
            / "predictions.jsonl"
        )

    def generation_evaluation(self, artifact: ArtifactRef, evaluation: str) -> Path:
        return (
            self.root
            / "generation"
            / artifact.name
            / artifact.revision
            / "evaluation"
            / evaluation
        )

    def stage_marker(self, fingerprint: str) -> Path:
        return self.root / "state" / "completed" / f"{fingerprint}.json"


class JsonlCheckpoint:
    """Append-only JSONL that resumes safely after an interrupted final write."""

    def __init__(self, path: str | Path, *, key: str = "task_id") -> None:
        self.path = Path(path)
        self.key = key
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records = self._load()

    @property
    def completed(self) -> set[str]:
        return set(self.records)

    def append(self, record: Mapping[str, Any]) -> None:
        self.append_many((record,))

    def append_many(self, records: Sequence[Mapping[str, Any]]) -> None:
        batch = [(cast(str, record[self.key]), dict(record)) for record in records]
        keys = [key for key, _ in batch]
        if len(keys) != len(set(keys)) or any(key in self.records for key in keys):
            raise ValueError("duplicate checkpoint key")
        if not batch:
            return

        payload = "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for _, record in batch
        ).encode("utf-8")
        with self.path.open("ab") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        self.records.update(batch)

    def _load(self) -> dict[str, dict[str, Any]]:
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
                with self.path.open("ab") as stream:
                    stream.write(b"\n")
                    stream.flush()
                    os.fsync(stream.fileno())

        rows = read_jsonl(self.path)
        records = {cast(str, row[self.key]): row for row in rows}
        if len(records) != len(rows):
            raise ValueError(f"duplicate checkpoint key: {self.path}")
        return records


class CandidateStore(JsonlCheckpoint):
    """Resume-safe candidates with views for retrieval and generation."""

    def append_hits(
        self,
        tasks: Mapping[str, BenchmarkTask],
        results: Mapping[str, Sequence[SearchHit]],
    ) -> None:
        self.append_many(
            [
                RankedCandidates(tasks[task_id], tuple(hits)).as_record()
                for task_id, hits in results.items()
            ]
        )

    def rankings(
        self,
        tasks: Mapping[str, BenchmarkTask],
    ) -> dict[str, RankedCandidates]:
        return {
            task_id: RankedCandidates.from_record(tasks[task_id], record)
            for task_id, record in self.records.items()
        }

    def contexts(self, top_k: int) -> dict[str, list[Context]]:
        result: dict[str, list[Context]] = {}
        for task_id, record in self.records.items():
            contexts = []
            for item in record.get("contexts", ()):
                text = str(item.get("text") or "")
                title = item.get("title")
                if not text.strip() and not (title or "").strip():
                    continue
                contexts.append(
                    Context(
                        document_id=str(item["document_id"]),
                        text=text,
                        title=title,
                        score=float(
                            item.get("retriever_score", item.get("score", 0.0))
                        ),
                    )
                )
            result[task_id] = contexts[:top_k]
        return result

    def write_prediction(
        self,
        path: Path,
        *,
        top_k: int,
        tasks: Sequence[BenchmarkTask],
    ) -> int:
        source = [
            self.records.get(task.task_id, task.as_record())
            for task in tasks
        ]
        predictions = (
            make_retrieval_record(
                record,
                [
                    {"document_id": context["document_id"]}
                    for context in record.get("contexts", ())
                ],
                max_contexts=top_k,
            )
            for record in source
        )
        return write_retrieval_jsonl(path, predictions)

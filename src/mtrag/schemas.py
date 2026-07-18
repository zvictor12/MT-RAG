from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class QueryVariant(StrEnum):
    """Text used as the retrieval query in an experiment."""

    LAST = "last"
    QWEN = "qwen"
    GOLD = "gold"


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    name: str
    revision: str


@dataclass(frozen=True, slots=True)
class Message:
    speaker: str
    text: str


@dataclass(frozen=True, slots=True)
class Context:
    document_id: str
    text: str
    title: str | None = None
    url: str | None = None
    score: float | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkTask:
    task_id: str
    conversation_id: str
    turn: int
    collection: str
    domain: str
    messages: tuple[Message, ...]
    contexts: tuple[Context, ...] = ()
    targets: tuple[Message, ...] = ()
    answerability: tuple[str, ...] = ()

    @property
    def final_question(self) -> str:
        for message in reversed(self.messages):
            if message.speaker == "user":
                return message.text
        raise ValueError("A benchmark task must contain a user message")

    @property
    def history(self) -> tuple[Message, ...]:
        for index in range(len(self.messages) - 1, -1, -1):
            if self.messages[index].speaker == "user":
                return self.messages[:index]
        raise ValueError("A benchmark task must contain a user message")

    def as_record(self, *, include_targets: bool = False) -> dict[str, Any]:
        record: dict[str, Any] = {
            "conversation_id": self.conversation_id,
            "task_id": self.task_id,
            "turn": self.turn,
            "Collection": self.collection,
            "input": [
                {"speaker": message.speaker, "text": message.text}
                for message in self.messages
            ],
        }
        if include_targets:
            record["targets"] = [
                {"speaker": message.speaker, "text": message.text}
                for message in self.targets
            ]
            record["Answerability"] = list(self.answerability)
        return record


@dataclass(frozen=True, slots=True)
class QueryCase:
    task_id: str
    domain: str
    variant: QueryVariant
    text: str


@dataclass(frozen=True, slots=True)
class BgeFeatures:
    dense: tuple[float, ...]
    sparse: dict[str, float]


@dataclass(frozen=True, slots=True)
class SearchQuery:
    task_id: str
    domain: str
    text: str
    bge: BgeFeatures | None = None


@dataclass(frozen=True, slots=True)
class SearchHit:
    document_id: str
    score: float
    rank: int
    source: str
    title: str | None = None
    text: str | None = None
    components: dict[str, float] = field(default_factory=dict)

    @property
    def has_passage(self) -> bool:
        return bool((self.title or "").strip() or (self.text or "").strip())


@dataclass(frozen=True, slots=True)
class RankedCandidates:
    task: BenchmarkTask
    hits: tuple[SearchHit, ...]

    def as_record(self) -> dict[str, Any]:
        record = self.task.as_record()
        record["contexts"] = []
        for hit in self.hits:
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
            record["contexts"].append(context)
        return record

    @classmethod
    def from_record(
        cls,
        task: BenchmarkTask,
        record: Mapping[str, Any],
    ) -> "RankedCandidates":
        hits = tuple(
            SearchHit(
                document_id=str(context["document_id"]),
                score=float(
                    context.get("retriever_score", context.get("score", 0.0))
                ),
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
        )
        return cls(task, hits)

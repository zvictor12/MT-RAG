import re
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

from mtrag.data.jsonl import iter_jsonl
from mtrag.schemas import (
    BenchmarkTask,
    Context,
    Message,
    QueryCase,
    QueryVariant,
)


DOMAINS = ("clapnq", "cloud", "govt", "fiqa")

COLLECTION_TO_DOMAIN = {
    "mt-rag-clapnq-elser-512-100-20240503": "clapnq",
    "mt-rag-ibmcloud-elser-512-100-20240502": "cloud",
    "mt-rag-govt-elser-512-100-20240611": "govt",
    "mt-rag-fiqa-beir-elser-512-100-20240501": "fiqa",
}

_CANONICAL_TASK_ID = re.compile(r"^(?P<conversation>.+)<::>(?P<turn>\d+)$")
_LEGACY_TASK_ID = re.compile(r"^(?P<conversation>.+)::(?P<turn>\d+)$")
_USER_PREFIX = re.compile(r"^\s*\|user\|\s*:\s*", re.IGNORECASE)


def normalize_task_id(task_id: str) -> str:
    """Convert the evaluator's legacy ``::turn`` suffix without blind replace."""
    value = task_id.strip()
    if not value:
        raise ValueError("task_id must not be empty")

    canonical = _CANONICAL_TASK_ID.fullmatch(value)
    if canonical is not None:
        return value

    legacy = _LEGACY_TASK_ID.fullmatch(value)
    if legacy is None:
        return value
    return f"{legacy['conversation']}<::>{legacy['turn']}"


def domain_for_collection(collection: str) -> str:
    try:
        return COLLECTION_TO_DOMAIN[collection]
    except KeyError as error:
        raise ValueError(f"Unknown MT-RAG collection: {collection!r}") from error


def clean_query_text(text: str) -> str:
    """Remove the BEIR role marker shared by gold and last-turn files."""
    return _USER_PREFIX.sub("", text, count=1).strip()


class BenchmarkRepository:
    """Read the canonical human MT-RAG tasks and retrieval query variants."""

    def __init__(self, root: str | Path = "../mt-rag-benchmark") -> None:
        self.root = Path(root)
        self.human_root = self.root / "mtrag-human"

    @property
    def generation_tasks_path(self) -> Path:
        return self.human_root / "generation_tasks" / "reference.jsonl"

    @property
    def retrieval_tasks_root(self) -> Path:
        return self.human_root / "retrieval_tasks"

    def load_tasks(self) -> tuple[BenchmarkTask, ...]:
        return tuple(
            self._parse_task(row)
            for row in iter_jsonl(self.generation_tasks_path)
        )

    def tasks_by_id(self) -> dict[str, BenchmarkTask]:
        tasks: dict[str, BenchmarkTask] = {}
        for task in self.load_tasks():
            if task.task_id in tasks:
                raise ValueError(f"Duplicate benchmark task_id: {task.task_id}")
            tasks[task.task_id] = task
        return tasks

    def query_cases(
        self,
        variant: QueryVariant | str,
        *,
        qwen_queries: Mapping[str, str] | str | PathLike[str] | None = None,
    ) -> tuple[QueryCase, ...]:
        selected = QueryVariant(variant)
        generated_variants = {
            QueryVariant.QWEN,
            QueryVariant.QWEN_T0,
            QueryVariant.QWEN_T02,
        }
        if selected in generated_variants:
            if qwen_queries is None:
                raise ValueError(
                    f"qwen_queries is required for the {selected.value} variant"
                )
            return self._qwen_query_cases(qwen_queries, selected)
        if qwen_queries is not None:
            raise ValueError("qwen_queries is only valid for generated variants")

        suffix = {
            QueryVariant.LAST: "lastturn",
            QueryVariant.GOLD: "rewrite",
        }[selected]
        cases: list[QueryCase] = []
        for domain in DOMAINS:
            path = self.retrieval_tasks_root / domain / f"{domain}_{suffix}.jsonl"
            for row in iter_jsonl(path):
                cases.append(
                    QueryCase(
                        task_id=normalize_task_id(_required_str(row, "_id")),
                        domain=domain,
                        variant=selected,
                        text=clean_query_text(_required_str(row, "text")),
                    )
                )
        return tuple(cases)

    def _qwen_query_cases(
        self,
        source: Mapping[str, str] | str | PathLike[str],
        variant: QueryVariant = QueryVariant.QWEN,
    ) -> tuple[QueryCase, ...]:
        queries = self._read_qwen_queries(source)
        tasks = self.tasks_by_id()
        unknown = queries.keys() - tasks.keys()
        if unknown:
            example = min(unknown)
            raise ValueError(f"Qwen query has unknown task_id: {example}")

        return tuple(
            QueryCase(
                task_id=task.task_id,
                domain=task.domain,
                variant=variant,
                text=clean_query_text(queries[task.task_id]),
            )
            for task in tasks.values()
            if task.task_id in queries
        )

    @staticmethod
    def _read_qwen_queries(
        source: Mapping[str, str] | str | PathLike[str],
    ) -> dict[str, str]:
        if isinstance(source, Mapping):
            pairs = source.items()
        else:
            pairs = (
                (
                    _required_str(row, "task_id"),
                    _required_str(row, "query"),
                )
                for row in iter_jsonl(Path(source))
            )

        queries: dict[str, str] = {}
        for raw_task_id, query in pairs:
            task_id = normalize_task_id(str(raw_task_id))
            if task_id in queries:
                raise ValueError(f"Duplicate Qwen task_id: {task_id}")
            text = str(query).strip()
            if not text:
                raise ValueError(f"Empty Qwen query for task_id: {task_id}")
            queries[task_id] = text
        return queries

    @staticmethod
    def _parse_task(row: Mapping[str, Any]) -> BenchmarkTask:
        collection = _required_str(row, "Collection")
        return BenchmarkTask(
            task_id=normalize_task_id(_required_str(row, "task_id")),
            conversation_id=_required_str(row, "conversation_id"),
            turn=int(row["turn"]),
            collection=collection,
            domain=domain_for_collection(collection),
            messages=tuple(_parse_message(item) for item in row.get("input", ())),
            contexts=tuple(_parse_context(item) for item in row.get("contexts", ())),
            targets=tuple(_parse_message(item) for item in row.get("targets", ())),
            answerability=tuple(row.get("Answerability", ())),
        )


def _required_str(row: Mapping[str, Any], field: str) -> str:
    try:
        value = row[field]
    except KeyError as error:
        raise ValueError(f"Missing required field: {field}") from error
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected a non-empty string in field: {field}")
    return value


def _parse_message(row: Mapping[str, Any]) -> Message:
    return Message(
        speaker=_required_str(row, "speaker"),
        text=_required_str(row, "text"),
    )


def _parse_context(row: Mapping[str, Any]) -> Context:
    score = row.get("score")
    return Context(
        document_id=_required_str(row, "document_id"),
        text=_required_str(row, "text"),
        title=row.get("title"),
        url=row.get("url"),
        score=float(score) if score is not None else None,
    )

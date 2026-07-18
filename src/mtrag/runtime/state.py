from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Mapping


MANIFEST_VERSION = 1


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"
    INTERRUPTED = "interrupted"


@dataclass(slots=True)
class StageState:
    status: StageStatus = StageStatus.PENDING
    attempts: int = 0
    started_at: str | None = None
    finished_at: str | None = None
    return_code: int | None = None
    pid: int | None = None
    error: str | None = None

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> StageState:
        fields = {**asdict(cls()), **value}
        fields["status"] = StageStatus(fields["status"])
        return cls(**fields)


@dataclass(slots=True)
class RunManifest:
    version: int = field(default=MANIFEST_VERSION, init=False)
    created_at: str
    updated_at: str
    stages: dict[str, StageState] = field(default_factory=dict)

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> RunManifest:
        version = int(value.get("version", 0))
        if version != MANIFEST_VERSION:
            raise ValueError(
                f"unsupported manifest version {version}; expected {MANIFEST_VERSION}"
            )
        return cls(
            created_at=str(value["created_at"]),
            updated_at=str(value["updated_at"]),
            stages={
                name: StageState.from_json(stage)
                for name, stage in value.get("stages", {}).items()
            },
        )

    def complete_for(self, stage_names: Iterable[str]) -> bool:
        complete = {StageStatus.SUCCEEDED, StageStatus.SKIPPED}
        return all(
            name in self.stages and self.stages[name].status in complete
            for name in stage_names
        )


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Replace a JSON file without ever exposing a partial document."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


class EventLog:
    """Append scheduler events as one durable JSON object per line."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(
        self,
        event: str,
        *,
        stage: str | None = None,
        **details: Any,
    ) -> None:
        record: dict[str, Any] = {"timestamp": utc_now(), "event": event}
        if stage is not None:
            record["stage"] = stage
        if details:
            record["details"] = details
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"

        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())


class StateStore:
    """Persist scheduler state and preserve completed stages across schedules."""

    def __init__(
        self,
        run_dir: Path,
        stage_names: Iterable[str],
        *,
        resume: bool = True,
    ) -> None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = run_dir / "manifest.json"
        self.events = EventLog(run_dir / "events.jsonl")
        names = tuple(stage_names)

        if resume and self.manifest_path.exists():
            document = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            self.manifest = RunManifest.from_json(document)
            self._resume(names)
        else:
            timestamp = utc_now()
            self.manifest = RunManifest(
                created_at=timestamp,
                updated_at=timestamp,
                stages={name: StageState() for name in names},
            )
            self.save()
            self.events.append("run_created", stage_count=len(names))

    def _resume(self, stage_names: Iterable[str]) -> None:
        reconsidered: list[str] = []
        for name in stage_names:
            state = self.manifest.stages.setdefault(name, StageState())
            if state.status is StageStatus.PENDING:
                continue
            state.status = StageStatus.PENDING
            state.finished_at = state.return_code = state.pid = state.error = None
            reconsidered.append(name)

        self.save()
        self.events.append("run_resumed", reconsidered_stages=reconsidered)

    def save(self) -> None:
        self.manifest.updated_at = utc_now()
        document = asdict(self.manifest)
        document["stages"] = dict(sorted(document["stages"].items()))
        write_json_atomic(self.manifest_path, document)

    def transition(
        self,
        name: str,
        status: StageStatus,
        *,
        return_code: int | None = None,
        pid: int | None = None,
        error: str | None = None,
    ) -> StageState:
        state = self.manifest.stages[name]
        timestamp = utc_now()

        if status is StageStatus.RUNNING:
            state.attempts += 1
            state.started_at = timestamp
            state.finished_at = None
        elif status is not StageStatus.PENDING:
            state.finished_at = timestamp

        state.status = status
        state.return_code = return_code
        state.pid = pid
        state.error = error
        self.save()
        self.events.append(
            f"stage_{status.value}",
            stage=name,
            attempt=state.attempts,
            return_code=return_code,
            error=error,
        )
        return state

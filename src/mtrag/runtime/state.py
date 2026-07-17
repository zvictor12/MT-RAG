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


@dataclass
class StageState:
    status: StageStatus = StageStatus.PENDING
    attempts: int = 0
    started_at: str | None = None
    finished_at: str | None = None
    return_code: int | None = None
    pid: int | None = None
    error: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StageState":
        return cls(
            status=StageStatus(value.get("status", StageStatus.PENDING)),
            attempts=int(value.get("attempts", 0)),
            started_at=value.get("started_at"),
            finished_at=value.get("finished_at"),
            return_code=value.get("return_code"),
            pid=value.get("pid"),
            error=value.get("error"),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        return value


@dataclass
class RunManifest:
    created_at: str
    updated_at: str
    stages: dict[str, StageState] = field(default_factory=dict)
    version: int = MANIFEST_VERSION

    @classmethod
    def empty(cls, stage_names: Iterable[str]) -> "RunManifest":
        timestamp = utc_now()
        return cls(
            created_at=timestamp,
            updated_at=timestamp,
            stages={name: StageState() for name in stage_names},
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RunManifest":
        version = int(value.get("version", 0))
        if version != MANIFEST_VERSION:
            raise ValueError(
                f"unsupported manifest version {version}; "
                f"expected {MANIFEST_VERSION}"
            )
        return cls(
            version=version,
            created_at=str(value["created_at"]),
            updated_at=str(value["updated_at"]),
            stages={
                name: StageState.from_dict(stage)
                for name, stage in value.get("stages", {}).items()
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "stages": {
                name: stage.to_dict()
                for name, stage in sorted(self.stages.items())
            },
        }

    @property
    def complete(self) -> bool:
        return all(
            stage.status in {StageStatus.SUCCEEDED, StageStatus.SKIPPED}
            for stage in self.stages.values()
        )


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Replace a JSON file without exposing a partially written manifest."""

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
    """Append-only JSONL audit log for scheduler decisions."""

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
        record: dict[str, Any] = {
            "timestamp": utc_now(),
            "event": event,
        }
        if stage is not None:
            record["stage"] = stage
        if details:
            record["details"] = details
        payload = (
            json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        with self._lock:
            descriptor = os.open(
                self.path,
                os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                0o644,
            )
            try:
                os.write(descriptor, payload)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


class StateStore:
    """Owns the run manifest and makes every state transition durable."""

    def __init__(
        self,
        run_dir: Path,
        stage_names: Iterable[str],
        *,
        resume: bool = True,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.manifest_path = self.run_dir / "manifest.json"
        self.events = EventLog(self.run_dir / "events.jsonl")
        self.run_dir.mkdir(parents=True, exist_ok=True)

        names = tuple(stage_names)
        if self.manifest_path.exists() and resume:
            self.manifest = RunManifest.from_dict(
                json.loads(self.manifest_path.read_text(encoding="utf-8"))
            )
            self._prepare_resume(names)
        else:
            self.manifest = RunManifest.empty(names)
            self.save()
            self.events.append("run_created", stage_count=len(names))

    def _prepare_resume(self, stage_names: Iterable[str]) -> None:
        names = tuple(stage_names)
        current = set(names)
        self.manifest.stages = {
            name: state
            for name, state in self.manifest.stages.items()
            if name in current
        }
        reset_statuses = {
            StageStatus.RUNNING,
            StageStatus.FAILED,
            StageStatus.BLOCKED,
            StageStatus.INTERRUPTED,
        }
        reset: list[str] = []
        for name in names:
            state = self.manifest.stages.setdefault(name, StageState())
            if state.status in reset_statuses:
                state.status = StageStatus.PENDING
                state.pid = None
                state.return_code = None
                state.finished_at = None
                state.error = None
                reset.append(name)
        self.save()
        self.events.append("run_resumed", reset_stages=reset)

    def save(self) -> None:
        self.manifest.updated_at = utc_now()
        write_json_atomic(self.manifest_path, self.manifest.to_dict())

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

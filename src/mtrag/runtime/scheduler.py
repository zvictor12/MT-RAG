from __future__ import annotations

import asyncio
import os
import signal
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

from mtrag.runtime.state import RunManifest, StageStatus, StateStore, utc_now
from mtrag.runtime.thermal import ThermalGuard


DONE = {StageStatus.SUCCEEDED, StageStatus.SKIPPED}
FAILED = {StageStatus.FAILED, StageStatus.BLOCKED}


@dataclass(frozen=True, slots=True)
class ResourceRequest:
    cpu_slots: int = 1
    gpu: bool = False


@dataclass(frozen=True, slots=True)
class StageSpec:
    name: str
    command: tuple[str, ...]
    dependencies: tuple[str, ...] = ()
    resources: ResourceRequest = field(default_factory=ResourceRequest)
    cwd: Path | None = None
    max_attempts: int = 1
    retry_delay: float = 1.0


class _ResourcePool:
    def __init__(self, cpu_slots: int) -> None:
        self.available = cpu_slots
        self.gpu_available = True

    def try_acquire(self, request: ResourceRequest) -> bool:
        if request.cpu_slots > self.available:
            return False
        if request.gpu and not self.gpu_available:
            return False
        self.available -= request.cpu_slots
        if request.gpu:
            self.gpu_available = False
        return True

    def release(self, request: ResourceRequest) -> None:
        self.available += request.cpu_slots
        if request.gpu:
            self.gpu_available = True


def _stage_map(stages: Sequence[StageSpec]) -> dict[str, StageSpec]:
    by_name = {stage.name: stage for stage in stages}
    if len(by_name) != len(stages):
        raise ValueError("stage names must be unique")

    known = set(by_name)
    for stage in stages:
        unknown = set(stage.dependencies) - known
        if unknown:
            raise ValueError(
                f"stage {stage.name!r} has unknown dependencies: "
                f"{', '.join(sorted(unknown))}"
            )
    pending = {name: set(stage.dependencies) for name, stage in by_name.items()}
    while pending:
        ready = {name for name, dependencies in pending.items() if not dependencies}
        if not ready:
            raise ValueError(f"dependency cycle contains stage {min(pending)!r}")
        pending = {
            name: dependencies - ready
            for name, dependencies in pending.items()
            if name not in ready
        }
    return by_name


class SubprocessScheduler:
    """Run a durable DAG with bounded CPU use and one exclusive GPU slot."""

    def __init__(
        self,
        stages: Sequence[StageSpec],
        run_dir: Path,
        *,
        cpu_slots: int | None = None,
        thermal_guard: ThermalGuard | None = None,
        resume: bool = True,
    ) -> None:
        self.stages = tuple(stages)
        self.by_name = _stage_map(self.stages)
        slots = cpu_slots if cpu_slots is not None else max(1, os.cpu_count() or 1)
        self.resources = _ResourcePool(slots)
        oversized = next(
            (stage for stage in self.stages if stage.resources.cpu_slots > slots), None
        )
        if oversized:
            raise ValueError(
                f"stage {oversized.name!r} requests more than {slots} CPU slots"
            )

        self.store = StateStore(Path(run_dir), self.by_name, resume=resume)
        self.logs_dir = Path(run_dir) / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.thermal_guard = thermal_guard
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._stop_requested = False
        self._force_requested = False

    @property
    def manifest(self) -> RunManifest:
        return self.store.manifest

    def request_stop(self, *, force: bool = False) -> None:
        """Interrupt children gracefully; a second request kills them."""
        force = force or self._stop_requested
        if (force and self._force_requested) or (not force and self._stop_requested):
            return
        self._stop_requested = True
        self._force_requested = force
        self.store.events.append("shutdown_forced" if force else "shutdown_requested")
        sig = signal.SIGKILL if force else signal.SIGINT
        for process in tuple(self._processes.values()):
            self._signal_process(process, sig)

    @staticmethod
    def _signal_process(
        process: asyncio.subprocess.Process,
        sig: signal.Signals,
    ) -> None:
        if process.returncode is not None:
            return
        try:
            os.killpg(process.pid, sig)
        except (ProcessLookupError, PermissionError):
            with suppress(ProcessLookupError, PermissionError):
                process.send_signal(sig)

    def _install_signal_handlers(self, loop: asyncio.AbstractEventLoop):
        installed = []
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.request_stop)
            except (NotImplementedError, RuntimeError, ValueError):
                continue
            installed.append(sig)
        return tuple(installed)

    def _schedule_ready(self) -> bool:
        changed = False
        for stage in self.stages:
            state = self.manifest.stages[stage.name]
            if state.status is not StageStatus.PENDING:
                continue
            dependencies = [
                self.manifest.stages[name].status for name in stage.dependencies
            ]
            if any(status in FAILED for status in dependencies):
                self.store.transition(
                    stage.name, StageStatus.BLOCKED, error="a dependency failed"
                )
                changed = True
                continue
            if not all(status in DONE for status in dependencies):
                continue
            if not self.resources.try_acquire(stage.resources):
                continue
            task = asyncio.create_task(self._execute(stage), name=f"stage:{stage.name}")
            self._tasks[stage.name] = task
            changed = True
        return changed

    async def _execute(self, stage: StageSpec) -> None:
        try:
            if self.thermal_guard is not None:
                resource = "gpu" if stage.resources.gpu else "cpu"
                await asyncio.to_thread(self.thermal_guard.wait, resource)
            if self._stop_requested:
                return

            for attempt in range(1, stage.max_attempts + 1):
                if self._stop_requested:
                    self.store.transition(
                        stage.name,
                        StageStatus.INTERRUPTED,
                        error="scheduler shutdown interrupted the stage",
                    )
                    return

                return_code = await self._run_once(stage)
                if self._stop_requested:
                    self.store.transition(
                        stage.name,
                        StageStatus.INTERRUPTED,
                        return_code=return_code,
                        error="scheduler shutdown interrupted the stage",
                    )
                    return
                if return_code == 0:
                    self.store.transition(
                        stage.name, StageStatus.SUCCEEDED, return_code=0
                    )
                    return

                self.store.transition(
                    stage.name,
                    StageStatus.FAILED,
                    return_code=return_code,
                    error=f"command exited with status {return_code}",
                )
                if attempt < stage.max_attempts:
                    self.store.events.append(
                        "stage_retrying",
                        stage=stage.name,
                        next_attempt=attempt + 1,
                        return_code=return_code,
                    )
                    await asyncio.sleep(stage.retry_delay)
        finally:
            self.resources.release(stage.resources)

    async def _run_once(self, stage: StageSpec) -> int:
        log_path = self.logs_dir / f"{stage.name}.log"
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{utc_now()}] $ {' '.join(stage.command)}\n")
            log.flush()
            self.store.transition(stage.name, StageStatus.RUNNING)
            try:
                process = await asyncio.create_subprocess_exec(
                    *stage.command,
                    cwd=stage.cwd,
                    stdout=log,
                    stderr=asyncio.subprocess.STDOUT,
                    start_new_session=True,
                )
            except (OSError, ValueError) as error:
                log.write(f"failed to start process: {error}\n")
                return 127

            self._processes[stage.name] = process
            self.manifest.stages[stage.name].pid = process.pid
            self.store.save()
            self.store.events.append(
                "stage_process_started", stage=stage.name, pid=process.pid
            )
            if self._stop_requested:
                sig = signal.SIGKILL if self._force_requested else signal.SIGINT
                self._signal_process(process, sig)
            try:
                while process.returncode is None:
                    await asyncio.sleep(0.1)
                return await process.wait()
            finally:
                self._processes.pop(stage.name, None)

    async def run(self) -> RunManifest:
        loop = asyncio.get_running_loop()
        handlers = self._install_signal_handlers(loop)
        self.store.events.append("scheduler_started")
        try:
            while True:
                changed = False if self._stop_requested else self._schedule_ready()
                if not self._tasks:
                    pending = any(
                        self.manifest.stages[name].status is StageStatus.PENDING
                        for name in self.by_name
                    )
                    if self._stop_requested or not pending:
                        break
                    if not changed:
                        raise RuntimeError(
                            "scheduler made no progress; "
                            "check dependencies and resources"
                        )
                    continue

                done, _ = await asyncio.wait(
                    self._tasks.values(),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for name, task in tuple(self._tasks.items()):
                    if task not in done:
                        continue
                    del self._tasks[name]
                    try:
                        task.result()
                    except asyncio.CancelledError:
                        if self.manifest.stages[name].status is StageStatus.RUNNING:
                            self.store.transition(
                                name,
                                StageStatus.INTERRUPTED,
                                error="scheduler task was cancelled",
                            )
                    except Exception as error:
                        self.store.transition(
                            name,
                            StageStatus.FAILED,
                            error=f"scheduler error: {type(error).__name__}: {error}",
                        )
        finally:
            for sig in handlers:
                loop.remove_signal_handler(sig)
            self.store.events.append(
                "scheduler_stopped", complete=self.manifest.complete_for(self.by_name),
                stop_requested=self._stop_requested,
            )
        return self.manifest

    def run_sync(self) -> RunManifest:
        return asyncio.run(self.run())

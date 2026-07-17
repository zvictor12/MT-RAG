from __future__ import annotations

import asyncio
import os
import signal
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from mtrag.runtime.state import RunManifest, StageStatus, StateStore, utc_now
from mtrag.runtime.thermal import ThermalGuard


StageCondition = Callable[[RunManifest], bool]


@dataclass(frozen=True)
class ResourceRequest:
    cpu_slots: int = 1
    gpu: bool = False

    def __post_init__(self) -> None:
        if self.cpu_slots < 1:
            raise ValueError("cpu_slots must be at least one")


@dataclass(frozen=True)
class StageSpec:
    name: str
    command: Sequence[str]
    dependencies: Sequence[str] = ()
    resources: ResourceRequest = field(default_factory=ResourceRequest)
    condition: StageCondition | None = None
    cwd: Path | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    max_attempts: int = 1
    retry_delay: float = 1.0

    def __post_init__(self) -> None:
        if not self.name or any(character.isspace() for character in self.name):
            raise ValueError("stage name must be non-empty and contain no whitespace")
        if not self.command:
            raise ValueError(f"stage {self.name!r} has an empty command")
        if self.name in self.dependencies:
            raise ValueError(f"stage {self.name!r} cannot depend on itself")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        if self.retry_delay < 0:
            raise ValueError("retry_delay cannot be negative")
        object.__setattr__(self, "command", tuple(str(part) for part in self.command))
        object.__setattr__(self, "dependencies", tuple(self.dependencies))
        if self.cwd is not None:
            object.__setattr__(self, "cwd", Path(self.cwd))


class _ResourcePool:
    def __init__(self, cpu_slots: int) -> None:
        if cpu_slots < 1:
            raise ValueError("cpu_slots must be at least one")
        self.capacity = cpu_slots
        self.available = cpu_slots
        self.gpu_available = True

    def validate(self, request: ResourceRequest) -> None:
        if request.cpu_slots > self.capacity:
            raise ValueError(
                f"stage requests {request.cpu_slots} CPU slots, but only "
                f"{self.capacity} are configured"
            )

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
        if self.available > self.capacity:
            raise RuntimeError("released more CPU slots than were acquired")
        if request.gpu:
            if self.gpu_available:
                raise RuntimeError("released a GPU slot that was not acquired")
            self.gpu_available = True


class SubprocessScheduler:
    """Run a small dependency graph with durable state and resource limits."""

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
        self.by_name = {stage.name: stage for stage in self.stages}
        if len(self.by_name) != len(self.stages):
            raise ValueError("stage names must be unique")
        self._validate_graph()

        slots = (
            cpu_slots
            if cpu_slots is not None
            else max(1, os.cpu_count() or 1)
        )
        self.resources = _ResourcePool(slots)
        for stage in self.stages:
            self.resources.validate(stage.resources)

        self.store = StateStore(
            Path(run_dir),
            self.by_name,
            resume=resume,
        )
        self.logs_dir = Path(run_dir) / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.thermal_guard = thermal_guard
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._interrupted_stages: set[str] = set()
        self._stop_requested = False
        self._force_requested = False
        self._signal_count = 0

    @property
    def manifest(self) -> RunManifest:
        return self.store.manifest

    def _validate_graph(self) -> None:
        known = set(self.by_name)
        for stage in self.stages:
            unknown = set(stage.dependencies) - known
            if unknown:
                names = ", ".join(sorted(unknown))
                raise ValueError(f"stage {stage.name!r} has unknown dependencies: {names}")

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visiting:
                raise ValueError(f"dependency cycle contains stage {name!r}")
            if name in visited:
                return
            visiting.add(name)
            for dependency in self.by_name[name].dependencies:
                visit(dependency)
            visiting.remove(name)
            visited.add(name)

        for name in self.by_name:
            visit(name)

    def request_stop(self, *, force: bool = False) -> None:
        """Stop scheduling work; a second request immediately kills children."""

        self._signal_count += 1
        force = force or self._signal_count > 1
        if force:
            if self._force_requested:
                return
            self._force_requested = True
            self._stop_requested = True
            self._interrupted_stages.update(
                name
                for name, process in self._processes.items()
                if process.returncode is None
            )
            self.store.events.append("shutdown_forced")
            self._signal_processes(signal.SIGKILL)
            return

        if self._stop_requested:
            return
        self._stop_requested = True
        self._interrupted_stages.update(
            name
            for name, process in self._processes.items()
            if process.returncode is None
        )
        self.store.events.append("shutdown_requested")
        self._signal_processes(signal.SIGINT)

    def _signal_processes(self, sig: signal.Signals) -> None:
        for process in tuple(self._processes.values()):
            if process.returncode is not None:
                continue
            try:
                os.killpg(process.pid, sig)
            except (ProcessLookupError, PermissionError):
                try:
                    process.send_signal(sig)
                except ProcessLookupError:
                    pass

    def _install_signal_handlers(
        self,
        loop: asyncio.AbstractEventLoop,
    ) -> dict[signal.Signals, object]:
        if threading.current_thread() is not threading.main_thread():
            return {}
        previous: dict[signal.Signals, object] = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                previous[sig] = signal.getsignal(sig)
                loop.add_signal_handler(sig, self.request_stop)
            except (NotImplementedError, RuntimeError, ValueError):
                continue
        return previous

    @staticmethod
    def _restore_signal_handlers(
        loop: asyncio.AbstractEventLoop,
        previous: Mapping[signal.Signals, object],
    ) -> None:
        for sig, handler in previous.items():
            loop.remove_signal_handler(sig)
            signal.signal(sig, handler)

    def _dependencies_succeeded(self, stage: StageSpec) -> bool:
        return all(
            self.manifest.stages[name].status
            in {StageStatus.SUCCEEDED, StageStatus.SKIPPED}
            for name in stage.dependencies
        )

    def _dependency_failed(self, stage: StageSpec) -> bool:
        return any(
            self.manifest.stages[name].status
            in {StageStatus.FAILED, StageStatus.BLOCKED}
            for name in stage.dependencies
        )

    def _mark_ready_stages(self) -> bool:
        """Schedule every ready stage that currently fits the resource pool."""

        changed = False
        for stage in self.stages:
            state = self.manifest.stages[stage.name]
            if state.status is not StageStatus.PENDING:
                continue
            if self._dependency_failed(stage):
                self.store.transition(
                    stage.name,
                    StageStatus.BLOCKED,
                    error="a dependency failed",
                )
                changed = True
                continue
            if not self._dependencies_succeeded(stage):
                continue
            if stage.condition is not None:
                try:
                    enabled = bool(stage.condition(self.manifest))
                except Exception as error:
                    self.store.transition(
                        stage.name,
                        StageStatus.FAILED,
                        error=f"condition raised {type(error).__name__}: {error}",
                    )
                    changed = True
                    continue
                if not enabled:
                    self.store.transition(stage.name, StageStatus.SKIPPED)
                    changed = True
                    continue
            if not self.resources.try_acquire(stage.resources):
                continue
            self._tasks[stage.name] = asyncio.create_task(
                self._run_with_resources(stage),
                name=f"stage:{stage.name}",
            )
            changed = True
        return changed

    async def _run_with_resources(self, stage: StageSpec) -> None:
        try:
            await self._run_stage(stage)
        finally:
            self.resources.release(stage.resources)

    async def _wait_for_temperature(self, stage: StageSpec) -> None:
        if self.thermal_guard is None:
            return
        resource = "gpu" if stage.resources.gpu else "cpu"
        await asyncio.to_thread(self.thermal_guard.wait, resource)

    async def _run_stage(self, stage: StageSpec) -> None:
        await self._wait_for_temperature(stage)
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
            if stage.name in self._interrupted_stages:
                self.store.transition(
                    stage.name,
                    StageStatus.INTERRUPTED,
                    return_code=return_code,
                    error="scheduler shutdown interrupted the stage",
                )
                return
            if return_code == 0:
                self.store.transition(
                    stage.name,
                    StageStatus.SUCCEEDED,
                    return_code=return_code,
                )
                return
            if self._stop_requested:
                self.store.transition(
                    stage.name,
                    StageStatus.INTERRUPTED,
                    return_code=return_code,
                    error="scheduler shutdown interrupted the stage",
                )
                return
            if attempt < stage.max_attempts:
                self.store.transition(
                    stage.name,
                    StageStatus.FAILED,
                    return_code=return_code,
                    error=f"command exited with status {return_code}",
                )
                self.store.events.append(
                    "stage_retrying",
                    stage=stage.name,
                    next_attempt=attempt + 1,
                    return_code=return_code,
                )
                await asyncio.sleep(stage.retry_delay)
                continue
            self.store.transition(
                stage.name,
                StageStatus.FAILED,
                return_code=return_code,
                error=f"command exited with status {return_code}",
            )

    async def _run_once(self, stage: StageSpec) -> int:
        environment = os.environ.copy()
        environment.update(stage.env)
        log_path = self.logs_dir / f"{stage.name}.log"
        log: TextIO
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{utc_now()}] $ {' '.join(stage.command)}\n")
            log.flush()
            self.store.transition(stage.name, StageStatus.RUNNING)
            try:
                process = await asyncio.create_subprocess_exec(
                    *stage.command,
                    cwd=stage.cwd,
                    env=environment,
                    stdout=log,
                    stderr=asyncio.subprocess.STDOUT,
                    start_new_session=True,
                )
            except (OSError, ValueError) as error:
                log.write(f"failed to start process: {error}\n")
                return 127

            self._processes[stage.name] = process
            state = self.manifest.stages[stage.name]
            state.pid = process.pid
            self.store.save()
            self.store.events.append(
                "stage_process_started",
                stage=stage.name,
                pid=process.pid,
            )
            if self._stop_requested:
                self._interrupted_stages.add(stage.name)
                try:
                    os.killpg(process.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
            try:
                # Periodic wakeups make child reaping reliable in constrained
                # notebook/container runtimes where SIGCHLD can be coalesced.
                while process.returncode is None:
                    await asyncio.sleep(0.1)
                return await process.wait()
            finally:
                self._processes.pop(stage.name, None)

    async def run(self) -> RunManifest:
        loop = asyncio.get_running_loop()
        previous_handlers = self._install_signal_handlers(loop)
        self.store.events.append("scheduler_started")
        try:
            while True:
                changed = False
                if not self._stop_requested:
                    changed = self._mark_ready_stages()

                if not self._tasks:
                    pending = any(
                        state.status is StageStatus.PENDING
                        for state in self.manifest.stages.values()
                    )
                    if not pending or self._stop_requested:
                        break
                    if not changed:
                        raise RuntimeError(
                            "scheduler made no progress; check dependencies and resources"
                        )

                if self._tasks:
                    done, _ = await asyncio.wait(
                        self._tasks.values(),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in done:
                        name = next(
                            stage_name
                            for stage_name, candidate in self._tasks.items()
                            if candidate is task
                        )
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
            self._restore_signal_handlers(loop, previous_handlers)
            self.store.events.append(
                "scheduler_stopped",
                complete=self.manifest.complete,
                stop_requested=self._stop_requested,
            )
        return self.manifest

    def run_sync(self) -> RunManifest:
        return asyncio.run(self.run())

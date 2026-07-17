import argparse
import fcntl
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from mtrag.experiments.artifacts import RunArtifacts
from mtrag.experiments.common import thermal_guard
from mtrag.experiments.planning import build_plan, build_workflow
from mtrag.experiments.preflight import preflight
from mtrag.experiments.reporting import render_experiment_results
from mtrag.experiments.spec import ExperimentConfig
from mtrag.experiments.stages import run_stage
from mtrag.runtime import SubprocessScheduler


DEFAULT_CONFIG = Path("configs/experiment.toml")


def _shared(
    parser: argparse.ArgumentParser,
    *,
    include_schedule: bool = False,
) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path)
    if include_schedule:
        parser.add_argument("--schedule", default="bge")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and resume config-declared MT-RAG experiments."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="print a configured schedule")
    _shared(plan, include_schedule=True)

    check = commands.add_parser("preflight", help="validate local services and models")
    _shared(check, include_schedule=True)

    run = commands.add_parser("run", help="run or resume a configured schedule")
    _shared(run, include_schedule=True)

    status = commands.add_parser("status", help="show one schedule's stage state")
    _shared(status, include_schedule=True)

    results = commands.add_parser("results", help="print every stored metric report")
    _shared(results)

    stage = commands.add_parser("stage", help="run one scheduler stage")
    stage.add_argument("name")
    _shared(stage, include_schedule=True)
    return parser.parse_args()


def _load(args: argparse.Namespace) -> tuple[ExperimentConfig, Path]:
    config = ExperimentConfig.load(args.config)
    run_dir = (args.run_dir or config.default_run_dir).expanduser().resolve()
    return config, run_dir


@contextmanager
def _config_snapshot(path: Path) -> Iterator[Path]:
    source = path.expanduser().resolve()
    descriptor, temporary_name = tempfile.mkstemp(
        dir=source.parent,
        prefix=f".{source.stem}.",
        suffix=".snapshot.toml",
    )
    snapshot = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(source.read_bytes())
        yield snapshot
    finally:
        snapshot.unlink(missing_ok=True)


@contextmanager
def _campaign_lock(run_dir: Path) -> Iterator[None]:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / ".scheduler.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another scheduler is already using {run_dir}"
            ) from error
        yield


def print_plan(config: ExperimentConfig, run_dir: Path, *, schedule: str) -> None:
    for stage in build_plan(config, run_dir, schedule=schedule):
        dependencies = ", ".join(stage.dependencies) or "-"
        resource = "GPU" if stage.resources.gpu else "CPU"
        print(f"{stage.name}  after={dependencies}  {resource}/{stage.resources.cpu_slots}")


def print_status(run_dir: Path, stage_names: set[str] | None = None) -> None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"run has not started: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shown = 0
    for name, state in manifest["stages"].items():
        if stage_names is not None and name not in stage_names:
            continue
        shown += 1
        print(
            f"{name}  {state['status']}  attempts={state.get('attempts', 0)}"
        )
    if shown == 0:
        print("no recorded stages for this schedule")


def main() -> None:
    args = parse_args()
    if args.command == "run":
        with _config_snapshot(args.config) as snapshot:
            args.config = snapshot
            _run(args)
        return

    config, run_dir = _load(args)

    if args.command == "plan":
        print_plan(config, run_dir, schedule=args.schedule)
        return
    if args.command == "preflight":
        workflow = build_workflow(config, schedule=args.schedule)
        preflight(config, RunArtifacts(run_dir), stages=workflow.stages)
        return
    if args.command == "status":
        plan = build_plan(config, run_dir, schedule=args.schedule)
        print_status(run_dir, {stage.name for stage in plan})
        return
    if args.command == "results":
        print(render_experiment_results(run_dir))
        return
    if args.command == "stage":
        run_stage(
            args.name,
            config,
            RunArtifacts(run_dir),
            schedule=args.schedule,
        )
        return


def _run(args: argparse.Namespace) -> None:
    config, run_dir = _load(args)
    with _campaign_lock(run_dir):
        print(f"run directory: {run_dir}", flush=True)
        print(f"stage logs: {run_dir / 'logs'}", flush=True)
        workflow = build_workflow(config, schedule=args.schedule)
        preflight(config, RunArtifacts(run_dir), stages=workflow.stages)
        plan = build_plan(config, run_dir, schedule=args.schedule)
        scheduler = SubprocessScheduler(
            plan,
            run_dir,
            cpu_slots=config.run.cpu_slots,
            thermal_guard=thermal_guard(config),
            resume=True,
        )
        manifest = scheduler.run_sync()
        print_status(run_dir, {stage.name for stage in plan})
        if not manifest.complete_for(stage.name for stage in plan):
            raise SystemExit(1)


if __name__ == "__main__":
    main()

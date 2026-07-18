import argparse
import fcntl
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from mtrag.experiments.artifacts import RunArtifacts
from mtrag.experiments.common import thermal_guard
from mtrag.experiments.planning import Workflow, build_plan, build_workflow
from mtrag.experiments.preflight import preflight
from mtrag.experiments.reporting import render_experiment_results
from mtrag.experiments.spec import ExperimentConfig
from mtrag.experiments.stages import run_stage
from mtrag.runtime import SubprocessScheduler


DEFAULT_CONFIG = Path("configs/experiment.toml")


def _shared(parser: argparse.ArgumentParser, *, schedule: bool = False) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path)
    if schedule:
        parser.add_argument("--schedule", default="bge")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and resume config-declared MT-RAG experiments."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("plan", "print a configured schedule"),
        ("preflight", "validate local services and models"),
        ("run", "run or resume a configured schedule"),
        ("status", "show one schedule's stage state"),
    ):
        _shared(commands.add_parser(name, help=help_text), schedule=True)
    _shared(commands.add_parser("results", help="print stored metric reports"))
    stage = commands.add_parser("stage", help=argparse.SUPPRESS)
    stage.add_argument("name")
    stage.add_argument("--plan", type=Path, required=True)
    _shared(stage)
    return parser.parse_args()


@contextmanager
def config_snapshot(path: Path) -> Iterator[Path]:
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
def campaign_lock(run_dir: Path) -> Iterator[None]:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / ".scheduler.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another scheduler is using {run_dir}") from error
        yield


@dataclass(frozen=True, slots=True)
class ExperimentApp:
    config: ExperimentConfig
    run_dir: Path

    @classmethod
    def load(cls, config_path: Path, run_dir: Path | None) -> "ExperimentApp":
        config = ExperimentConfig.load(config_path)
        root = (run_dir or config.default_run_dir).expanduser().resolve()
        return cls(config, root)

    def show_plan(self, schedule: str) -> None:
        for stage in build_workflow(self.config, schedule=schedule).stages:
            dependencies = ", ".join(stage.dependencies) or "-"
            resource = "GPU" if stage.gpu else "CPU"
            print(
                f"{stage.name}  after={dependencies}  "
                f"{resource}/{stage.cpu_slots}"
            )

    def show_status(self, workflow: Workflow) -> int:
        path = self.run_dir / "manifest.json"
        if not path.exists():
            print(f"run has not started: {path}")
            return 1
        states = json.loads(path.read_text(encoding="utf-8"))["stages"]
        shown = [stage.name for stage in workflow.stages if stage.name in states]
        for name in shown:
            state = states[name]
            print(f"{name}  {state['status']}  attempts={state.get('attempts', 0)}")
        if not shown:
            print("no recorded stages for this schedule")
        return 0

    def check(self, schedule: str) -> None:
        workflow = build_workflow(self.config, schedule=schedule)
        preflight(self.config, stages=workflow.stages)

    def run(self, schedule: str) -> int:
        with campaign_lock(self.run_dir):
            workflow = build_workflow(self.config, schedule=schedule)
            print(f"run directory: {self.run_dir}", flush=True)
            print(f"stage logs: {self.run_dir / 'logs'}", flush=True)
            preflight(self.config, stages=workflow.stages)
            plan_path = workflow.save(self.run_dir)
            plan = build_plan(
                self.config,
                self.run_dir,
                workflow=workflow,
                plan_path=plan_path,
            )
            manifest = SubprocessScheduler(
                plan,
                self.run_dir,
                cpu_slots=self.config.run.cpu_slots,
                thermal_guard=thermal_guard(self.config),
                resume=True,
            ).run_sync()
            self.show_status(workflow)
            return 0 if manifest.complete_for(stage.name for stage in plan) else 1


def main() -> int:
    args = parse_args()
    if args.command == "run":
        with config_snapshot(args.config) as snapshot:
            return ExperimentApp.load(snapshot, args.run_dir).run(args.schedule)

    app = ExperimentApp.load(args.config, args.run_dir)
    if args.command == "plan":
        app.show_plan(args.schedule)
    elif args.command == "preflight":
        app.check(args.schedule)
    elif args.command == "status":
        workflow = build_workflow(app.config, schedule=args.schedule)
        return app.show_status(workflow)
    elif args.command == "results":
        print(render_experiment_results(app.run_dir))
    else:
        run_stage(
            args.name,
            app.config,
            RunArtifacts(app.run_dir),
            workflow=Workflow.load(args.plan),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

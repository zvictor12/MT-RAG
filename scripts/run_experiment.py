import argparse
import copy
import hashlib
import json
import shutil
import tomllib
from pathlib import Path

from mtrag.experiments.artifacts import RunArtifacts, lock_run_definition
from mtrag.experiments.common import thermal_guard
from mtrag.experiments.planning import PHASES, build_plan
from mtrag.experiments.reporting import render_experiment_results
from mtrag.experiments.spec import ExperimentConfig
from mtrag.experiments.stages import STAGES, run_stage
from mtrag.llm.prompts import GENERATOR_PROMPT_VERSION, REWRITE_PROMPT_VERSION
from mtrag.runtime import SubprocessScheduler
from mtrag.runtime.state import write_json_atomic


DEFAULT_CONFIG = Path("configs/experiment.toml")
CONFIG_DIGEST_SCOPE = "excluding-thermal-v1"


def _shared(parser: argparse.ArgumentParser, *, include_phase: bool = False) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path)
    if include_phase:
        parser.add_argument("--phase", choices=PHASES, default="bge")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and resume the local MT-RAG experiment DAG."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="print the stage graph")
    _shared(plan, include_phase=True)

    preflight = commands.add_parser(
        "preflight",
        help="validate models, CUDA, services, and restored BGE indices",
    )
    _shared(preflight)

    run = commands.add_parser("run", help="run or resume all stages")
    _shared(run, include_phase=True)

    status = commands.add_parser("status", help="show durable stage state")
    _shared(status)

    results = commands.add_parser("results", help="print experiment metric summaries")
    _shared(results)

    stage = commands.add_parser(
        "stage",
        help="run one stage (used internally by the scheduler)",
    )
    stage.add_argument("name", choices=tuple(STAGES))
    _shared(stage)
    return parser.parse_args()


def _load(args: argparse.Namespace) -> tuple[ExperimentConfig, Path]:
    config = ExperimentConfig.load(args.config)
    run_dir = (args.run_dir or config.default_run_dir).expanduser().resolve()
    return config, run_dir


def print_plan(config: ExperimentConfig, run_dir: Path, *, phase: str) -> None:
    for spec in build_plan(config, run_dir, phase=phase):
        dependencies = ", ".join(spec.dependencies) or "-"
        resource = "GPU" if spec.resources.gpu else "CPU"
        print(
            f"{spec.name:24} after={dependencies:36} "
            f"resource={resource}/{spec.resources.cpu_slots}"
        )


def print_status(run_dir: Path) -> None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"run has not started: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, state in manifest["stages"].items():
        attempts = state.get("attempts", 0)
        print(f"{name:24} {state['status']:12} attempts={attempts}")


def _config_digest(path: Path) -> str:
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    return _config_document_digest(document)


def _config_document_digest(document: dict) -> str:
    document = copy.deepcopy(document)
    document.pop("thermal", None)
    canonical = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _run_definition(config: ExperimentConfig) -> dict[str, str]:
    return {
        "config_sha256": _config_digest(config.path),
        "config_digest_scope": CONFIG_DIGEST_SCOPE,
        "rewrite_prompt_version": REWRITE_PROMPT_VERSION,
        "generator_prompt_version": GENERATOR_PROMPT_VERSION,
    }


def _migrate_legacy_definition(
    artifacts: RunArtifacts,
    definition: dict[str, str],
) -> None:
    if not artifacts.definition.exists() or not artifacts.config_snapshot.exists():
        return
    existing = json.loads(artifacts.definition.read_text(encoding="utf-8"))
    legacy = {
        "config_sha256": hashlib.sha256(
            artifacts.config_snapshot.read_bytes()
        ).hexdigest(),
        "rewrite_prompt_version": REWRITE_PROMPT_VERSION,
        "generator_prompt_version": GENERATOR_PROMPT_VERSION,
    }
    if existing != legacy:
        return
    if _config_digest(artifacts.config_snapshot) != definition["config_sha256"]:
        return
    write_json_atomic(artifacts.definition, definition)


def _migrate_rewrite_variants_definition(
    artifacts: RunArtifacts,
    config: ExperimentConfig,
    definition: dict[str, str],
) -> None:
    """Extend a locked legacy run only with the checksummed rewrite matrix."""

    if not artifacts.definition.exists() or not artifacts.config_snapshot.exists():
        return
    existing = json.loads(artifacts.definition.read_text(encoding="utf-8"))
    current = tomllib.loads(config.path.read_text(encoding="utf-8"))
    snapshot = tomllib.loads(
        artifacts.config_snapshot.read_text(encoding="utf-8")
    )

    rewriting = current.get("rewriting")
    if not isinstance(rewriting, dict) or "variants" not in rewriting:
        return
    if config.rewriting.variant("qwen_t0").temperature != 0.0:
        return

    legacy_current = copy.deepcopy(current)
    legacy_rewriting = legacy_current.get("rewriting")
    if not isinstance(legacy_rewriting, dict):
        return
    legacy_rewriting.pop("variants", None)
    if not legacy_rewriting:
        legacy_current.pop("rewriting", None)

    comparable_current = copy.deepcopy(legacy_current)
    comparable_snapshot = copy.deepcopy(snapshot)
    comparable_current.pop("thermal", None)
    comparable_snapshot.pop("thermal", None)
    if comparable_current != comparable_snapshot:
        return

    legacy_definition = {
        "config_sha256": _config_document_digest(snapshot),
        "config_digest_scope": CONFIG_DIGEST_SCOPE,
        "rewrite_prompt_version": REWRITE_PROMPT_VERSION,
        "generator_prompt_version": GENERATOR_PROMPT_VERSION,
    }
    if existing != legacy_definition:
        return

    write_json_atomic(artifacts.definition, definition)
    shutil.copyfile(config.path, artifacts.config_snapshot)


def lock_definition(config: ExperimentConfig, run_dir: Path) -> None:
    artifacts = RunArtifacts(run_dir)
    definition = _run_definition(config)
    _migrate_legacy_definition(artifacts, definition)
    _migrate_rewrite_variants_definition(artifacts, config, definition)
    lock_run_definition(artifacts.definition, definition)
    if not artifacts.config_snapshot.exists():
        artifacts.config_snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(config.path, artifacts.config_snapshot)


def main() -> None:
    args = parse_args()
    config, run_dir = _load(args)

    if args.command == "plan":
        print_plan(config, run_dir, phase=args.phase)
        return
    if args.command == "preflight":
        run_stage("preflight", config, RunArtifacts(run_dir))
        return
    if args.command == "status":
        print_status(run_dir)
        return
    if args.command == "results":
        print(render_experiment_results(run_dir))
        return
    if args.command == "stage":
        run_stage(args.name, config, RunArtifacts(run_dir))
        return

    lock_definition(config, run_dir)
    print(f"run directory: {run_dir}", flush=True)
    print(f"stage logs: {run_dir / 'logs'}", flush=True)
    scheduler = SubprocessScheduler(
        build_plan(config, run_dir, phase=args.phase),
        run_dir,
        cpu_slots=config.run.cpu_slots,
        thermal_guard=thermal_guard(config),
        resume=True,
    )
    manifest = scheduler.run_sync()
    print_status(run_dir)
    if not manifest.complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

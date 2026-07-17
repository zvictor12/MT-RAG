from __future__ import annotations

import argparse
import filecmp
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

from mtrag.data import BenchmarkRepository
from mtrag.data.jsonl import read_jsonl
from mtrag.encoding import BgeFeatureStore
from mtrag.experiments.artifacts import RunArtifacts
from mtrag.experiments.planning import PlannedStage, Workflow, build_workflow
from mtrag.experiments.query_stages import query_cases
from mtrag.experiments.spec import ExperimentConfig
from mtrag.runtime.state import write_json_atomic
from mtrag.schemas import BgeFeatures


LEGACY_CANDIDATES = {
    "bge_dense_last": "bge_last.dense",
    "bge_sparse_last": "bge_last.sparse",
    "bge_rrf_last": "bge_last.rrf",
    "bge_rrf_last_reranked": "bge_last.rrf_reranked",
    "bge_dense_qwen": "bge_t0.dense",
    "bge_sparse_qwen": "bge_t0.sparse",
    "bge_rrf_qwen": "bge_t0.rrf",
    "bge_rrf_qwen_reranked": "bge_t0.rrf_reranked",
    "bge_dense_qwen_t02": "bge_t02.dense",
    "bge_sparse_qwen_t02": "bge_t02.sparse",
    "bge_rrf_qwen_t02": "bge_t02.rrf",
    "bge_rrf_qwen_t02_reranked": "bge_t02.rrf_reranked",
    "bge_dense_gold": "bge_gold.dense",
}
LEGACY_REWRITES = {
    "qwen_t0": ("qwen_t0.jsonl", "qwen.jsonl"),
    "qwen_t02": ("qwen_t02.jsonl",),
}
LEGACY_FEATURE_PREFIXES = {
    "last": "last",
    "gold": "gold",
    "qwen": "qwen_t0",
    "qwen_t02": "qwen_t02",
}
LEGACY_GENERATIONS = {
    "task_b": "task_b.jsonl",
    "task_c_bge_t0_legacy": "task_c_bge.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory or import artifacts produced by the legacy runner."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.toml"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/main"))
    parser.add_argument("--schedule", default="bge")
    parser.add_argument(
        "--trust-current-definition",
        action="store_true",
        help="register legacy files under the current experiment revisions",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.load(args.config)
    artifacts = RunArtifacts(args.run_dir.expanduser().resolve())
    workflow = build_workflow(config, schedule=args.schedule)

    _print_inventory(artifacts.root)
    if not args.trust_current_definition:
        print(
            "inventory only; pass --trust-current-definition only when the files "
            "match the current config, prompts and model revisions.",
            flush=True,
        )
        return

    task_ids = {
        task.task_id
        for task in BenchmarkRepository(config.run.benchmark_root).load_tasks()
    }
    _import_rewrites(artifacts, workflow, task_ids)
    _import_features(config, artifacts, workflow)
    _import_candidates(config, artifacts, workflow)
    _import_generations(artifacts, workflow, task_ids)


def _print_inventory(root: Path) -> None:
    files = {
        path
        for directory in ("candidates", "rewrites", "predictions")
        for path in (root / directory).glob("*.jsonl")
    }
    files.update(
        path
        for path in (
            root / "features" / "bge" / "dense.npz",
            root / "features" / "bge" / "sparse.jsonl",
        )
        if path.is_file()
    )
    if not files:
        print("legacy inventory: no files found", flush=True)
        return
    print("legacy inventory:", flush=True)
    for path in sorted(files):
        print(f"  {path.relative_to(root)}", flush=True)


def _import_rewrites(
    artifacts: RunArtifacts,
    workflow: Workflow,
    expected_ids: set[str],
) -> None:
    for stage in workflow.stages:
        if stage.kind != "rewrite":
            continue
        query_name = stage.params["query_name"]
        source = next(
            (
                artifacts.root / "rewrites" / filename
                for filename in LEGACY_REWRITES.get(query_name, ())
                if (artifacts.root / "rewrites" / filename).is_file()
            ),
            None,
        )
        if source is None:
            continue
        target = artifacts.rewrite(query_name, stage.params["query_revision"])
        _copy_jsonl(source, target, expected_ids, ("query",))
        _mark(artifacts, stage)


def _import_features(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
    workflow: Workflow,
) -> None:
    legacy = artifacts.root / "features" / "bge"
    paths = (legacy / "dense.npz", legacy / "sparse.jsonl")
    if not any(path.exists() for path in paths):
        return
    if not all(path.is_file() for path in paths):
        raise RuntimeError("legacy BGE feature store is incomplete")

    grouped: dict[str, dict[str, BgeFeatures]] = {}
    for key, value in BgeFeatureStore(legacy).load().items():
        prefix, separator, task_id = key.partition(":")
        query_name = LEGACY_FEATURE_PREFIXES.get(prefix)
        if separator and query_name:
            grouped.setdefault(query_name, {})[task_id] = value

    for stage in workflow.stages:
        if stage.kind != "encode":
            continue
        query_name = stage.params["query_name"]
        features = grouped.get(query_name)
        if not features:
            continue
        expected_ids = _query_ids(config, artifacts, workflow, query_name)
        if set(features) != expected_ids:
            raise RuntimeError(f"legacy BGE features do not match {query_name!r}")
        target = artifacts.bge_features(query_name, stage.params["feature_revision"])
        _save_features(features, target)
        _mark(artifacts, stage)


def _import_candidates(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
    workflow: Workflow,
) -> None:
    stages = {
        stage.params["reference"]: stage
        for stage in workflow.stages
        if stage.kind in {"retrieve", "fuse", "rerank"}
    }
    for legacy_name, reference in LEGACY_CANDIDATES.items():
        stage = stages.get(reference)
        source = artifacts.root / "candidates" / f"{legacy_name}.jsonl"
        if stage is None or not source.is_file():
            continue
        pipeline, _ = config.resolve_retrieval_output(reference)
        expected_ids = _query_ids(config, artifacts, workflow, pipeline.query)
        target = artifacts.candidates(reference, stage.params["revision"])
        _copy_jsonl(source, target, expected_ids, ("contexts",))
        _mark(artifacts, stage)


def _import_generations(
    artifacts: RunArtifacts,
    workflow: Workflow,
    expected_ids: set[str],
) -> None:
    stages = {
        stage.params["job_name"]: stage
        for stage in workflow.stages
        if stage.kind == "generate"
    }
    for job_name, filename in LEGACY_GENERATIONS.items():
        stage = stages.get(job_name)
        source = artifacts.root / "predictions" / filename
        if stage is None or not source.is_file():
            continue
        target = artifacts.generation(job_name, stage.params["revision"])
        _copy_jsonl(source, target, expected_ids, ("predictions",))
        _mark(artifacts, stage)


def _query_ids(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
    workflow: Workflow,
    query_name: str,
) -> set[str]:
    revision = next(
        stage.params["query_revision"]
        for stage in workflow.stages
        if stage.kind == "encode" and stage.params["query_name"] == query_name
    )
    return {
        case.task_id for case in query_cases(config, artifacts, query_name, revision)
    }


def _copy_jsonl(
    source: Path,
    target: Path,
    expected_ids: set[str],
    required_fields: Sequence[str],
) -> None:
    records = read_jsonl(source)
    task_ids = [record.get("task_id") for record in records]
    valid = (
        all(isinstance(task_id, str) for task_id in task_ids)
        and len(task_ids) == len(set(task_ids))
        and set(task_ids) == expected_ids
        and all(all(field in record for field in required_fields) for record in records)
    )
    if not valid:
        raise RuntimeError(f"legacy artifact is incomplete or incompatible: {source}")
    _copy_file(source, target)


def _copy_file(source: Path, target: Path) -> None:
    if target.exists():
        if not target.is_file() or not filecmp.cmp(source, target, shallow=False):
            raise RuntimeError(f"current revision target already differs: {target}")
        print(f"reused {target}", flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    print(f"copied {source} -> {target}", flush=True)


def _save_features(features: Mapping[str, BgeFeatures], target: Path) -> None:
    if target.exists():
        if BgeFeatureStore(target).load() != dict(features):
            raise RuntimeError(f"current feature revision already differs: {target}")
        print(f"reused {target}", flush=True)
        return
    BgeFeatureStore(target).save(features)
    print(f"copied legacy BGE features -> {target}", flush=True)


def _mark(artifacts: RunArtifacts, stage: PlannedStage) -> None:
    write_json_atomic(
        artifacts.stage_marker(stage.fingerprint),
        {"stage": stage.name, "imported_from_legacy": True},
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from mtrag.data import BenchmarkRepository
from mtrag.encoding import BgeFeatureStore
from mtrag.experiments.artifacts import RunArtifacts, read_jsonl
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
        help=(
            "assert that the legacy files were produced with the current config, "
            "model revisions and prompts, then register them under current fingerprints"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.load(args.config)
    artifacts = RunArtifacts(args.run_dir.expanduser().resolve())
    workflow = build_workflow(config, schedule=args.schedule)

    _print_inventory(artifacts, workflow)
    if not args.trust_current_definition:
        print(
            "inventory only; no files or completion markers were created. "
            "Re-run with --trust-current-definition only if these files match "
            "the current config, prompts and model revisions.",
            flush=True,
        )
        return

    _import_rewrites(config, artifacts, workflow)
    _import_features(config, artifacts, workflow)
    _import_candidates(config, artifacts, workflow)
    _import_task_b(config, artifacts, workflow)


def _print_inventory(artifacts: RunArtifacts, workflow: Workflow) -> None:
    selected = _selected_legacy_sources(artifacts, workflow)
    known = _known_legacy_sources(artifacts)
    discovered = set()
    for directory in ("candidates", "rewrites", "predictions"):
        discovered.update((artifacts.root / directory).glob("*.jsonl"))
    discovered.update(
        path
        for path in (
            artifacts.root / "features" / "bge" / "dense.npz",
            artifacts.root / "features" / "bge" / "sparse.jsonl",
        )
        if path.exists()
    )

    if not discovered:
        print("legacy inventory: no files found", flush=True)
        return
    print("legacy inventory:", flush=True)
    for path in sorted(discovered):
        if path in selected:
            status = "selected"
        elif path in known:
            status = "known"
        else:
            status = "ignored"
        print(f"  {status:8} {_display_path(artifacts.root, path)}", flush=True)


def _selected_legacy_sources(
    artifacts: RunArtifacts,
    workflow: Workflow,
) -> set[Path]:
    stages = workflow.stages
    outputs = {
        stage.params["reference"]
        for stage in stages
        if stage.kind in {"retrieve", "fuse", "rerank"}
    }
    selected = {
        artifacts.root / "candidates" / f"{old_name}.jsonl"
        for old_name, reference in LEGACY_CANDIDATES.items()
        if reference in outputs
    }
    rewrite_names = {
        stage.params["query_name"] for stage in stages if stage.kind == "rewrite"
    }
    for query_name in rewrite_names:
        selected.update(
            artifacts.root / "rewrites" / filename
            for filename in LEGACY_REWRITES.get(query_name, ())
        )
    if any(stage.kind == "encode" for stage in stages):
        selected.update(
            {
                artifacts.root / "features" / "bge" / "dense.npz",
                artifacts.root / "features" / "bge" / "sparse.jsonl",
            }
        )
    if any(
        stage.kind == "generate" and stage.params["job_name"] == "task_b"
        for stage in stages
    ):
        selected.add(artifacts.root / "predictions" / "task_b.jsonl")
    return selected


def _known_legacy_sources(artifacts: RunArtifacts) -> set[Path]:
    sources = {
        artifacts.root / "candidates" / f"{name}.jsonl"
        for name in LEGACY_CANDIDATES
    }
    sources.update(
        artifacts.root / "rewrites" / filename
        for filenames in LEGACY_REWRITES.values()
        for filename in filenames
    )
    sources.update(
        {
            artifacts.root / "features" / "bge" / "dense.npz",
            artifacts.root / "features" / "bge" / "sparse.jsonl",
            artifacts.root / "predictions" / "task_b.jsonl",
        }
    )
    return sources


def _import_rewrites(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
    workflow: Workflow,
) -> None:
    expected_ids = {
        task.task_id
        for task in BenchmarkRepository(config.run.benchmark_root).load_tasks()
    }
    for stage in workflow.stages:
        if stage.kind != "rewrite":
            continue
        query_name = stage.params["query_name"]
        source = _first_file(
            artifacts.root / "rewrites" / filename
            for filename in LEGACY_REWRITES.get(query_name, ())
        )
        if source is None:
            continue
        target = artifacts.rewrite(query_name, stage.params["query_revision"])
        _copy_jsonl(
            source,
            target,
            expected_ids=expected_ids,
            required_fields=("query",),
        )
        _mark(artifacts, stage, sources=(source,), targets=(target,))


def _import_features(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
    workflow: Workflow,
) -> None:
    legacy = artifacts.root / "features" / "bge"
    legacy_paths = (legacy / "dense.npz", legacy / "sparse.jsonl")
    if not any(path.exists() for path in legacy_paths):
        return
    if not all(path.is_file() for path in legacy_paths):
        raise RuntimeError("legacy BGE feature store is incomplete")

    features = BgeFeatureStore(legacy).load()
    grouped: dict[str, dict[str, BgeFeatures]] = {
        query: {} for query in LEGACY_FEATURE_PREFIXES.values()
    }
    for key, value in features.items():
        prefix, separator, task_id = key.partition(":")
        query_name = LEGACY_FEATURE_PREFIXES.get(prefix)
        if separator and query_name is not None:
            grouped[query_name][task_id] = value

    for stage in workflow.stages:
        if stage.kind != "encode":
            continue
        query_name = stage.params["query_name"]
        imported = grouped.get(query_name, {})
        if not imported:
            continue
        expected_ids = _query_ids(config, artifacts, workflow, query_name)
        _require_exact_ids(
            imported,
            expected_ids,
            f"legacy BGE features for {query_name}",
        )
        target = artifacts.bge_features(
            query_name,
            stage.params["feature_revision"],
        )
        _copy_feature_store(imported, target)
        target_paths = (target / "dense.npz", target / "sparse.jsonl")
        _mark(artifacts, stage, sources=legacy_paths, targets=target_paths)


def _import_candidates(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
    workflow: Workflow,
) -> None:
    stages = _stages_by_output(workflow.stages)
    for old_name, reference in LEGACY_CANDIDATES.items():
        stage = stages.get(reference)
        source = artifacts.root / "candidates" / f"{old_name}.jsonl"
        if stage is None or not source.is_file():
            continue
        pipeline, _output = config.resolve_retrieval_output(reference)
        expected_ids = _query_ids(config, artifacts, workflow, pipeline.query)
        target = artifacts.candidates(reference, stage.params["revision"])
        _copy_jsonl(
            source,
            target,
            expected_ids=expected_ids,
            required_fields=("contexts",),
        )
        _mark(artifacts, stage, sources=(source,), targets=(target,))


def _import_task_b(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
    workflow: Workflow,
) -> None:
    stage = next(
        (
            item
            for item in workflow.stages
            if item.kind == "generate" and item.params["job_name"] == "task_b"
        ),
        None,
    )
    source = artifacts.root / "predictions" / "task_b.jsonl"
    if stage is None or not source.is_file():
        return
    expected_ids = {
        task.task_id
        for task in BenchmarkRepository(config.run.benchmark_root).load_tasks()
    }
    target = artifacts.generation("task_b", stage.params["revision"])
    _copy_jsonl(
        source,
        target,
        expected_ids=expected_ids,
        required_fields=("predictions",),
    )
    _mark(artifacts, stage, sources=(source,), targets=(target,))


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
        case.task_id
        for case in query_cases(config, artifacts, query_name, revision)
    }


def _stages_by_output(stages: Sequence[PlannedStage]) -> dict[str, PlannedStage]:
    return {
        stage.params["reference"]: stage
        for stage in stages
        if stage.kind in {"retrieve", "fuse", "rerank"}
    }


def _first_file(paths: Iterable[Path]) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


def _copy_jsonl(
    source: Path,
    target: Path,
    *,
    expected_ids: set[str],
    required_fields: Sequence[str],
) -> None:
    _validate_jsonl(source, expected_ids, required_fields)
    _copy_file(source, target)
    _validate_jsonl(target, expected_ids, required_fields)
    if _sha256(source) != _sha256(target):
        raise RuntimeError(f"legacy copy differs from its source: {target}")


def _validate_jsonl(
    path: Path,
    expected_ids: set[str],
    required_fields: Sequence[str],
) -> None:
    records = read_jsonl(path)
    if not records:
        raise RuntimeError(f"legacy artifact is empty: {path}")
    ids: dict[str, None] = {}
    for record in records:
        task_id = record.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise RuntimeError(f"legacy artifact has no task_id: {path}")
        if task_id in ids:
            raise RuntimeError(
                f"legacy artifact has duplicate task_id {task_id!r}: {path}"
            )
        missing = [field for field in required_fields if field not in record]
        if missing:
            raise RuntimeError(f"legacy artifact is missing {missing[0]!r}: {path}")
        ids[task_id] = None
    _require_exact_ids(ids, expected_ids, str(path))


def _require_exact_ids(
    records: Mapping[str, object],
    expected_ids: set[str],
    label: str,
) -> None:
    actual_ids = set(records)
    if actual_ids == expected_ids:
        return
    missing = expected_ids - actual_ids
    extra = actual_ids - expected_ids
    raise RuntimeError(
        f"{label} is incomplete or incompatible: "
        f"expected={len(expected_ids)}, actual={len(actual_ids)}, "
        f"missing={len(missing)}, extra={len(extra)}"
    )


def _copy_file(source: Path, target: Path) -> None:
    if not source.is_file() or source.stat().st_size == 0:
        raise RuntimeError(f"legacy source is missing or empty: {source}")
    source_hash = _sha256(source)
    if target.exists():
        if not target.is_file() or _sha256(target) != source_hash:
            raise RuntimeError(f"current revision target already differs: {target}")
        _make_read_only(target)
        print(f"verified existing {target}", flush=True)
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copy2(source, temporary)
        if _sha256(temporary) != source_hash:
            raise RuntimeError(f"legacy copy verification failed: {target}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    _make_read_only(target)
    print(f"copied {source} -> {target}", flush=True)


def _copy_feature_store(
    features: Mapping[str, BgeFeatures],
    target: Path,
) -> None:
    if target.exists():
        if not target.is_dir() or BgeFeatureStore(target).load() != dict(features):
            raise RuntimeError(f"current feature revision already differs: {target}")
        _make_read_only(target / "dense.npz")
        _make_read_only(target / "sparse.jsonl")
        print(f"verified existing {target}", flush=True)
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        BgeFeatureStore(temporary).save(features)
        if BgeFeatureStore(temporary).load() != dict(features):
            raise RuntimeError(f"legacy feature copy verification failed: {target}")
        temporary.replace(target)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    if BgeFeatureStore(target).load() != dict(features):
        raise RuntimeError(f"legacy feature target is incomplete: {target}")
    _make_read_only(target / "dense.npz")
    _make_read_only(target / "sparse.jsonl")
    print(f"copied legacy BGE features -> {target}", flush=True)


def _mark(
    artifacts: RunArtifacts,
    stage: PlannedStage,
    *,
    sources: Sequence[Path],
    targets: Sequence[Path],
) -> None:
    for target in targets:
        if not target.is_file() or target.stat().st_size == 0:
            raise RuntimeError(f"refusing to mark incomplete target: {target}")
    write_json_atomic(
        artifacts.stage_marker(stage.fingerprint),
        {
            "stage": stage.name,
            "kind": stage.kind,
            "fingerprint": stage.fingerprint,
            "params": stage.params,
            "imported_from_legacy": True,
            "legacy_import": {
                "trust_current_definition": True,
                "sources": [_file_manifest(artifacts.root, path) for path in sources],
                "targets": [_file_manifest(artifacts.root, path) for path in targets],
            },
        },
    )


def _file_manifest(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": _display_path(root, path),
        "size": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _make_read_only(path: Path) -> None:
    path.chmod(path.stat().st_mode & ~0o222)


if __name__ == "__main__":
    main()

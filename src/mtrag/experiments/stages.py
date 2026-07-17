from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from mtrag.experiments.artifacts import RunArtifacts
from mtrag.experiments.generation_stages import (
    evaluate_generation_jobs,
    generate_job,
    unload_ollama,
)
from mtrag.experiments.planning import PlannedStage, build_workflow
from mtrag.experiments.query_stages import encode_bge_query, rewrite_query
from mtrag.experiments.retrieval_stages import (
    evaluate_task_a,
    fuse,
    rerank,
    retrieve,
)
from mtrag.experiments.spec import ExperimentConfig
from mtrag.runtime.state import write_json_atomic


Executor = Callable[[ExperimentConfig, RunArtifacts, Mapping[str, Any]], None]


def _call(function) -> Executor:
    return lambda config, artifacts, params: function(
        config,
        artifacts,
        **params,
    )


EXECUTORS: dict[str, Executor] = {
    "rewrite": _call(rewrite_query),
    "encode": _call(encode_bge_query),
    "retrieve": _call(retrieve),
    "fuse": _call(fuse),
    "rerank": _call(rerank),
    "evaluate_task_a": _call(evaluate_task_a),
    "generate": _call(generate_job),
    "unload_ollama": _call(unload_ollama),
    "evaluate_generation_batch": _call(evaluate_generation_jobs),
}


def run_stage(
    name: str,
    config: ExperimentConfig,
    artifacts: RunArtifacts,
    *,
    schedule: str,
) -> None:
    workflow = build_workflow(config, schedule=schedule)
    stage = workflow.stage(name)
    artifacts.create_directories()
    if _is_complete(stage, artifacts):
        print(f"reused {stage.name}", flush=True)
        return
    EXECUTORS[stage.kind](config, artifacts, stage.params)
    write_json_atomic(
        artifacts.stage_marker(stage.fingerprint),
        {
            "stage": stage.name,
            "kind": stage.kind,
            "fingerprint": stage.fingerprint,
            "params": stage.params,
        },
    )


def _is_complete(stage: PlannedStage, artifacts: RunArtifacts) -> bool:
    if stage.kind == "unload_ollama":
        return False
    marker = artifacts.stage_marker(stage.fingerprint)
    if not marker.is_file():
        return False
    try:
        stored = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        stored.get("stage") != stage.name
        or stored.get("kind") != stage.kind
        or stored.get("fingerprint") != stage.fingerprint
    ):
        return False
    params = stage.params
    if stage.kind == "rewrite":
        return artifacts.rewrite(
            params["query_name"], params["query_revision"]
        ).is_file()
    if stage.kind == "encode":
        directory = artifacts.bge_features(
            params["query_name"], params["feature_revision"]
        )
        return (directory / "dense.npz").is_file() and (
            directory / "sparse.jsonl"
        ).is_file()
    if stage.kind in {"retrieve", "fuse", "rerank"}:
        return artifacts.candidates(
            params["reference"], params["revision"]
        ).is_file()
    if stage.kind == "evaluate_task_a":
        return artifacts.retrieval_report(
            params["reference"],
            params["revision"],
            params["evaluation_revision"],
        ).is_file()
    if stage.kind == "generate":
        return artifacts.generation(
            params["job_name"], params["revision"]
        ).is_file()
    if stage.kind == "evaluate_generation_batch":
        return all(
            artifacts.generation_metrics(
                job["job_name"],
                job["generation_revision"],
                job["evaluation_revision"],
            ).is_file()
            and artifacts.generation_summary(
                job["job_name"],
                job["generation_revision"],
                job["evaluation_revision"],
            ).is_file()
            for job in params["jobs"]
        )
    return True

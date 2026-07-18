from __future__ import annotations

from collections.abc import Callable

from mtrag.experiments.artifacts import RunArtifacts
from mtrag.experiments.generation_stages import (
    evaluate_generation_jobs,
    generate_job,
    unload_ollama,
)
from mtrag.experiments.planning import Workflow
from mtrag.experiments.query_stages import encode_bge_query, rewrite_query
from mtrag.experiments.retrieval_stages import (
    evaluate_task_a,
    fuse,
    rerank,
    retrieve,
)
from mtrag.experiments.spec import ExperimentConfig
from mtrag.runtime.state import write_json_atomic


EXECUTORS: dict[str, Callable] = {
    "rewrite": rewrite_query,
    "encode": encode_bge_query,
    "retrieve": retrieve,
    "fuse": fuse,
    "rerank": rerank,
    "evaluate_task_a": evaluate_task_a,
    "generate": generate_job,
    "unload_ollama": unload_ollama,
    "evaluate_generation_batch": evaluate_generation_jobs,
}


def run_stage(
    name: str,
    config: ExperimentConfig,
    artifacts: RunArtifacts,
    *,
    workflow: Workflow,
) -> None:
    stage = workflow.stage(name)
    marker = artifacts.stage_marker(stage.fingerprint)
    if stage.kind != "unload_ollama" and marker.is_file():
        print(f"reused {stage.name}", flush=True)
        return
    EXECUTORS[stage.kind](config, artifacts, **stage.params)
    write_json_atomic(marker, {"stage": stage.name})

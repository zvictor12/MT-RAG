from __future__ import annotations

import sys
from pathlib import Path

from mtrag.experiments.artifacts import RunArtifacts
from mtrag.experiments.retrieval_stages import reranker_enabled
from mtrag.experiments.spec import ExperimentConfig
from mtrag.runtime import ResourceRequest, StageSpec


PHASES = ("bge", "full")


def build_plan(
    config: ExperimentConfig,
    run_dir: str | Path,
    *,
    phase: str = "bge",
) -> tuple[StageSpec, ...]:
    if phase not in PHASES:
        choices = ", ".join(PHASES)
        raise ValueError(
            f"unknown experiment phase {phase!r}; expected one of: {choices}"
        )

    root = Path(run_dir).expanduser().resolve()
    artifacts = RunArtifacts(root)
    script = config.project_root / "scripts" / "run_experiment.py"

    def command(stage: str) -> tuple[str, ...]:
        return (
            sys.executable,
            str(script),
            "stage",
            stage,
            "--config",
            str(config.path),
            "--run-dir",
            str(root),
        )

    def stage(
        name: str,
        *,
        dependencies: tuple[str, ...] = (),
        cpu_slots: int = 1,
        gpu: bool = False,
        condition=None,
        max_attempts: int = 1,
    ) -> StageSpec:
        return StageSpec(
            name=name,
            command=command(name),
            dependencies=dependencies,
            resources=ResourceRequest(cpu_slots=cpu_slots, gpu=gpu),
            condition=condition,
            cwd=config.project_root,
            max_attempts=max_attempts,
            retry_delay=5.0,
        )

    elasticsearch_slots = max(1, config.run.cpu_slots - 1)
    gate = lambda _manifest: reranker_enabled(artifacts)
    generation_metrics = lambda _manifest: (
        config.generation.run_algorithmic_metrics
    )
    legacy_bge_generation = lambda _manifest: not config.rewriting.variants
    selected_bge_generation = lambda _manifest: phase == "bge"
    selected_bge_metrics = lambda _manifest: (
        phase == "bge" and config.generation.run_algorithmic_metrics
    )
    task_b_dependencies = (
        ("select_bge_variants",)
        if phase == "bge"
        else ("select_winner",)
    )

    bge_plan = (
        stage("preflight"),
        stage("rewrite_qwen", dependencies=("preflight",), gpu=True),
        stage("encode_bge", dependencies=("rewrite_qwen",), gpu=True),
        stage(
            "retrieve_bge",
            dependencies=("encode_bge",),
            cpu_slots=elasticsearch_slots,
            max_attempts=2,
        ),
        stage("fuse_bge", dependencies=("retrieve_bge",)),
        stage(
            "evaluate_bge_base",
            dependencies=("fuse_bge",),
        ),
        stage("rerank_bge", dependencies=("fuse_bge",), gpu=True),
        stage(
            "evaluate_bge_rerank",
            dependencies=("rerank_bge",),
        ),
        stage(
            "decide_reranker",
            dependencies=("evaluate_bge_base", "evaluate_bge_rerank"),
        ),
        stage("select_bge", dependencies=("decide_reranker",)),
        stage(
            "generate_task_b",
            dependencies=task_b_dependencies,
            gpu=True,
        ),
        stage(
            "generate_task_c_bge",
            dependencies=("generate_task_b", "select_bge"),
            gpu=True,
            condition=legacy_bge_generation,
        ),
        stage(
            "evaluate_generation_bge",
            dependencies=("generate_task_c_bge",),
            gpu=True,
            condition=lambda manifest: (
                legacy_bge_generation(manifest)
                and generation_metrics(manifest)
            ),
        ),
        stage(
            "rewrite_qwen_t02",
            dependencies=("rewrite_qwen",),
            gpu=True,
        ),
        stage(
            "encode_bge_variants",
            dependencies=(
                "encode_bge",
                "retrieve_bge",
                "rewrite_qwen_t02",
            ),
            gpu=True,
        ),
        stage(
            "retrieve_bge_variants",
            dependencies=("encode_bge_variants",),
            cpu_slots=elasticsearch_slots,
            max_attempts=2,
        ),
        stage(
            "fuse_bge_variants",
            dependencies=("retrieve_bge", "retrieve_bge_variants"),
        ),
        stage(
            "evaluate_bge_variants_base",
            dependencies=("evaluate_bge_base", "fuse_bge_variants"),
        ),
        stage(
            "select_rewrite_variant",
            dependencies=("evaluate_bge_variants_base",),
        ),
        stage(
            "rerank_bge_variants",
            dependencies=("fuse_bge_variants",),
            gpu=True,
        ),
        stage(
            "evaluate_bge_variants_rerank",
            dependencies=("rerank_bge_variants",),
        ),
        stage(
            "decide_bge_variants",
            dependencies=(
                "evaluate_bge_base",
                "evaluate_bge_rerank",
                "evaluate_bge_variants_base",
                "evaluate_bge_variants_rerank",
            ),
        ),
        stage(
            "select_bge_variants",
            dependencies=("decide_bge_variants", "select_rewrite_variant"),
        ),
        stage(
            "generate_task_c_bge_last",
            dependencies=("rerank_bge_variants",),
            gpu=True,
            condition=selected_bge_generation,
        ),
        stage(
            "evaluate_generation_bge_last",
            dependencies=(
                "generate_task_c_bge_last",
                "select_bge_variants",
            ),
            gpu=True,
            condition=selected_bge_metrics,
        ),
        stage(
            "generate_task_c_bge_selected",
            dependencies=("generate_task_b", "select_bge_variants"),
            gpu=True,
            condition=selected_bge_generation,
        ),
        stage(
            "evaluate_generation_bge_selected",
            dependencies=("generate_task_c_bge_selected",),
            gpu=True,
            condition=selected_bge_metrics,
        ),
    )
    if phase == "bge":
        return bge_plan

    return bge_plan + (
        stage(
            "retrieve_elser",
            dependencies=("select_rewrite_variant",),
            cpu_slots=elasticsearch_slots,
            max_attempts=2,
        ),
        stage("evaluate_elser_base", dependencies=("retrieve_elser",)),
        stage(
            "rerank_elser",
            dependencies=("decide_bge_variants", "retrieve_elser"),
            gpu=True,
            condition=gate,
        ),
        stage(
            "evaluate_elser_rerank",
            dependencies=("rerank_elser",),
            condition=gate,
        ),
        stage(
            "select_winner",
            dependencies=(
                "select_bge_variants",
                "evaluate_elser_base",
                "evaluate_elser_rerank",
            ),
        ),
        stage(
            "generate_task_c",
            dependencies=("generate_task_b", "select_winner"),
            gpu=True,
        ),
        stage(
            "evaluate_generation",
            dependencies=("generate_task_c",),
            gpu=True,
            condition=generation_metrics,
        ),
    )

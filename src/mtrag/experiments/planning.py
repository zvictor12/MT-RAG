from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mtrag.data.benchmark import DOMAINS
from mtrag.experiments.fingerprint import file_sha256, fingerprint
from mtrag.experiments.spec import ExperimentConfig
from mtrag.runtime import ResourceRequest, StageSpec


SEMANTICS = {
    "rewrite": "ollama-rewrite-v1",
    "encode": "flagembedding-bge-m3-v1",
    "retrieve": "elasticsearch-retrieval-v1",
    "rrf": "rrf-v1",
    "rerank": "bge-v2-m3-rerank-v1",
    "generate": "ollama-grounded-generation-v1",
    "unload": "ollama-unload-v1",
}


@dataclass(frozen=True, slots=True)
class PlannedStage:
    name: str
    kind: str
    fingerprint: str
    params: dict[str, Any]
    dependencies: tuple[str, ...] = ()
    cpu_slots: int = 1
    gpu: bool = False
    max_attempts: int = 1


@dataclass(frozen=True, slots=True)
class Workflow:
    stages: tuple[PlannedStage, ...]

    def stage(self, name: str) -> PlannedStage:
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise ValueError(f"unknown planned stage: {name}")


@dataclass
class _Builder:
    config: ExperimentConfig
    stages: dict[str, PlannedStage] = field(default_factory=dict)
    queries: dict[str, tuple[str, str | None]] = field(default_factory=dict)
    features: dict[str, tuple[str, str]] = field(default_factory=dict)
    outputs: dict[str, tuple[str, str]] = field(default_factory=dict)
    generations: dict[str, tuple[str, str]] = field(default_factory=dict)
    generation_evaluations: dict[str, str] = field(default_factory=dict)

    def build(self, schedule_name: str) -> Workflow:
        schedule = self.config.schedule(schedule_name)
        for reference in schedule.task_a:
            self._task_a(reference)
        for job_name in schedule.generation:
            self._generation(job_name)
        self._finish_generation(schedule_name)
        return Workflow(stages=tuple(self.stages.values()))

    def _query(self, name: str) -> tuple[str, str | None]:
        if name in self.queries:
            return self.queries[name]
        query = self.config.query(name)
        inputs: dict[str, Any] = {
            "kind": query.kind,
            "sources": _query_sources(self.config, query.kind),
        }
        stage_name = None
        if query.kind == "rewrite":
            assert query.prompt is not None
            inputs.update(
                {
                    "prompt_sha256": file_sha256(query.prompt),
                    "temperature": query.temperature,
                    "max_tokens": query.max_tokens,
                    "model": self.config.models.ollama_model,
                    "model_digest": self.config.models.ollama_digest,
                    "num_ctx": self.config.models.ollama_num_ctx,
                    "seed": self.config.models.ollama_seed,
                    "semantics": SEMANTICS["rewrite"],
                }
            )
        revision = fingerprint("query", inputs)
        if query.kind == "rewrite":
            stage_name = self._add(
                "rewrite",
                name,
                revision,
                {"query_name": name, "query_revision": revision},
                gpu=True,
            )
        self.queries[name] = revision, stage_name
        return self.queries[name]

    def _features(self, query_name: str) -> tuple[str, str]:
        if query_name in self.features:
            return self.features[query_name]
        query_revision, query_stage = self._query(query_name)
        revision = fingerprint(
            "bge-query-features",
            {
                "query": query_revision,
                "model_revision": self.config.models.bge_revision,
                "max_length": self.config.models.bge_max_length,
                "semantics": SEMANTICS["encode"],
            },
        )
        dependencies = (query_stage,) if query_stage else ()
        stage_name = self._add(
            "encode",
            f"bge.{query_name}",
            revision,
            {
                "query_name": query_name,
                "query_revision": query_revision,
                "feature_revision": revision,
            },
            dependencies=dependencies,
            gpu=True,
        )
        self.features[query_name] = revision, stage_name
        return self.features[query_name]

    def _output(self, reference: str) -> tuple[str, str]:
        if reference in self.outputs:
            return self.outputs[reference]
        pipeline, output = self.config.resolve_retrieval_output(reference)
        query_revision, query_stage = self._query(pipeline.query)

        if pipeline.kind == "bge" and output in {"dense", "sparse"}:
            feature_revision, feature_stage = self._features(pipeline.query)
            revision = fingerprint(
                f"bge-{output}",
                {
                    "features": feature_revision,
                    "model_revision": self.config.models.bge_revision,
                    "index_revision": self.config.retrieval.bge_index_revision,
                    "top_k": getattr(self.config.retrieval, f"{output}_top_k"),
                    "candidate_multiplier": (
                        self.config.retrieval.dense_candidate_multiplier
                        if output == "dense"
                        else None
                    ),
                    "rescore_oversample": (
                        self.config.retrieval.dense_rescore_oversample
                        if output == "dense"
                        else None
                    ),
                    "semantics": SEMANTICS["retrieve"],
                },
            )
            stage_name = self._add(
                "retrieve",
                reference,
                revision,
                {
                    "reference": reference,
                    "revision": revision,
                    "query_name": pipeline.query,
                    "query_revision": query_revision,
                    "feature_revision": feature_revision,
                    "method": output,
                },
                dependencies=(feature_stage,),
                cpu_slots=max(1, self.config.run.cpu_slots - 1),
                max_attempts=2,
            )
        elif pipeline.kind == "bge" and output == "rrf":
            dense_reference = f"{pipeline.name}.dense"
            sparse_reference = f"{pipeline.name}.sparse"
            dense_revision, dense_stage = self._output(dense_reference)
            sparse_revision, sparse_stage = self._output(sparse_reference)
            revision = fingerprint(
                "rrf",
                {
                    "dense": dense_revision,
                    "sparse": sparse_revision,
                    "top_k": self.config.retrieval.rrf_top_k,
                    "rank_constant": self.config.retrieval.rrf_rank_constant,
                    "semantics": SEMANTICS["rrf"],
                },
            )
            stage_name = self._add(
                "fuse",
                reference,
                revision,
                {
                    "reference": reference,
                    "revision": revision,
                    "dense_reference": dense_reference,
                    "dense_revision": dense_revision,
                    "sparse_reference": sparse_reference,
                    "sparse_revision": sparse_revision,
                },
                dependencies=(dense_stage, sparse_stage),
            )
        elif pipeline.kind == "elser" and output == "base":
            revision = fingerprint(
                "elser",
                {
                    "query": query_revision,
                    "inference_id": self.config.services.elser_inference_id,
                    "index_revision": self.config.retrieval.elser_index_revision,
                    "top_k": self.config.retrieval.elser_top_k,
                    "semantics": SEMANTICS["retrieve"],
                },
            )
            stage_name = self._add(
                "retrieve",
                reference,
                revision,
                {
                    "reference": reference,
                    "revision": revision,
                    "query_name": pipeline.query,
                    "query_revision": query_revision,
                    "method": "elser",
                },
                dependencies=(query_stage,) if query_stage else (),
                cpu_slots=max(1, self.config.run.cpu_slots - 1),
                max_attempts=2,
            )
        else:
            source_output = "rrf" if pipeline.kind == "bge" else "base"
            source_reference = f"{pipeline.name}.{source_output}"
            source_revision, source_stage = self._output(source_reference)
            revision = fingerprint(
                "rerank",
                {
                    "source": source_revision,
                    "query": query_revision,
                    "model_revision": self.config.models.reranker_revision,
                    "max_length": self.config.models.reranker_max_length,
                    "input_top_k": self.config.reranking.input_top_k,
                    "output_top_k": self.config.reranking.output_top_k,
                    "semantics": SEMANTICS["rerank"],
                },
            )
            stage_name = self._add(
                "rerank",
                reference,
                revision,
                {
                    "reference": reference,
                    "revision": revision,
                    "source_reference": source_reference,
                    "source_revision": source_revision,
                    "query_name": pipeline.query,
                    "query_revision": query_revision,
                },
                dependencies=(source_stage,),
                gpu=True,
            )

        self.outputs[reference] = revision, stage_name
        return self.outputs[reference]

    def _task_a(self, reference: str) -> None:
        output_revision, output_stage = self._output(reference)
        revision = fingerprint(
            "ibm-task-a",
            {
                "retrieval": output_revision,
                "prediction_top_k": self.config.retrieval.prediction_top_k,
                "task_source": file_sha256(_generation_tasks(self.config)),
                "official_sources": _task_a_sources(self.config),
                "adapter_sources": _local_evaluation_sources("retrieval.py"),
            },
        )
        self._add(
            "evaluate_task_a",
            reference,
            revision,
            {
                "reference": reference,
                "revision": output_revision,
                "evaluation_revision": revision,
            },
            dependencies=(output_stage,),
        )

    def _generation(self, job_name: str) -> None:
        if job_name in self.generations:
            return
        job = self.config.generation_job(job_name)
        generator = self.config.generator(job.generator)
        dependencies: tuple[str, ...] = ()
        params: dict[str, Any] = {"job_name": job_name}
        task_revision = file_sha256(_generation_tasks(self.config))
        if job.contexts == "reference":
            context_revision = task_revision
        else:
            context_revision, context_stage = self._output(job.contexts)
            dependencies = (context_stage,)
            params.update(
                {
                    "context_reference": job.contexts,
                    "context_revision": context_revision,
                }
            )
        revision = fingerprint(
            "generation",
            {
                "task": job.task,
                "task_source": task_revision,
                "contexts": context_revision,
                "model": self.config.models.ollama_model,
                "model_digest": self.config.models.ollama_digest,
                "num_ctx": self.config.models.ollama_num_ctx,
                "seed": self.config.models.ollama_seed,
                "prompt_sha256": file_sha256(generator.prompt),
                "temperature": generator.temperature,
                "max_tokens": generator.max_tokens,
                "context_top_k": generator.context_top_k,
                "semantics": SEMANTICS["generate"],
            },
        )
        params["revision"] = revision
        stage_name = self._add(
            "generate",
            job_name,
            revision,
            params,
            dependencies=dependencies,
            gpu=True,
        )
        self.generations[job_name] = revision, stage_name
        if not job.evaluate:
            return
        evaluation_revision = fingerprint(
            "ibm-generation-evaluation",
            {
                "generation": revision,
                "official_sources": _generation_evaluation_sources(self.config),
                "bertscore_model": self.config.evaluation.bertscore_model,
                "adapter_sources": _local_evaluation_sources("generation.py"),
            },
        )
        self.generation_evaluations[job_name] = evaluation_revision

    def _finish_generation(self, schedule_name: str) -> None:
        if not self.generations:
            return

        generation_stages = tuple(stage for _, stage in self.generations.values())
        unload_revision = fingerprint(
            "ollama-unload",
            {
                "generations": sorted(
                    revision for revision, _ in self.generations.values()
                ),
                "model": self.config.models.ollama_model,
                "semantics": SEMANTICS["unload"],
            },
        )
        unload_stage = self._add(
            "unload_ollama",
            schedule_name,
            unload_revision,
            {},
            dependencies=generation_stages,
            gpu=True,
        )

        jobs = [
            {
                "job_name": job_name,
                "generation_revision": self.generations[job_name][0],
                "evaluation_revision": evaluation_revision,
            }
            for job_name, evaluation_revision in self.generation_evaluations.items()
        ]
        if not jobs:
            return
        batch_revision = fingerprint(
            "ibm-generation-evaluation-batch",
            {"evaluations": sorted(self.generation_evaluations.values())},
        )
        self._add(
            "evaluate_generation_batch",
            schedule_name,
            batch_revision,
            {"jobs": jobs},
            dependencies=(unload_stage,),
            gpu=True,
        )

    def _add(
        self,
        kind: str,
        logical_name: str,
        revision: str,
        params: dict[str, Any],
        *,
        dependencies: tuple[str, ...] = (),
        cpu_slots: int = 1,
        gpu: bool = False,
        max_attempts: int = 1,
    ) -> str:
        prefix = f"{kind}.{logical_name}" if logical_name else kind
        name = f"{prefix}.{revision[:12]}"
        stage = PlannedStage(
            name=name,
            kind=kind,
            fingerprint=revision,
            params=params,
            dependencies=dependencies,
            cpu_slots=cpu_slots,
            gpu=gpu,
            max_attempts=max_attempts,
        )
        previous = self.stages.setdefault(name, stage)
        if previous != stage:
            raise RuntimeError(f"stage fingerprint collision: {name}")
        return name


def build_workflow(config: ExperimentConfig, *, schedule: str) -> Workflow:
    return _Builder(config).build(schedule)


def build_plan(
    config: ExperimentConfig,
    run_dir: str | Path,
    *,
    schedule: str,
) -> tuple[StageSpec, ...]:
    root = Path(run_dir).expanduser().resolve()
    workflow = build_workflow(config, schedule=schedule)
    script = config.project_root / "scripts" / "run_experiment.py"
    return tuple(
        StageSpec(
            name=stage.name,
            command=(
                sys.executable,
                str(script),
                "stage",
                stage.name,
                "--schedule",
                schedule,
                "--config",
                str(config.path),
                "--run-dir",
                str(root),
            ),
            dependencies=stage.dependencies,
            resources=ResourceRequest(
                cpu_slots=stage.cpu_slots,
                gpu=stage.gpu,
            ),
            cwd=config.project_root,
            max_attempts=stage.max_attempts,
            retry_delay=5.0,
        )
        for stage in workflow.stages
    )


def _generation_tasks(config: ExperimentConfig) -> Path:
    return (
        config.run.benchmark_root
        / "mtrag-human"
        / "generation_tasks"
        / "reference.jsonl"
    )


def _query_sources(config: ExperimentConfig, kind: str) -> dict[str, str]:
    if kind == "rewrite":
        path = _generation_tasks(config)
        return _file_digests(config.run.benchmark_root, [path])
    suffix = "lastturn" if kind == "last_turn" else "rewrite"
    paths = [
        config.run.benchmark_root
        / "mtrag-human"
        / "retrieval_tasks"
        / domain
        / f"{domain}_{suffix}.jsonl"
        for domain in DOMAINS
    ]
    return _file_digests(config.run.benchmark_root, paths)


def _task_a_sources(config: ExperimentConfig) -> dict[str, str]:
    root = config.run.benchmark_root
    paths = [root / "scripts/evaluation/run_retrieval_eval.py"] + [
        root / "mtrag-human/retrieval_tasks" / domain / "qrels/dev.tsv"
        for domain in DOMAINS
    ]
    return _file_digests(root, paths)


def _generation_evaluation_sources(config: ExperimentConfig) -> dict[str, str]:
    root = config.run.benchmark_root / "scripts/evaluation"
    paths = [root / "run_algorithmic.py", root / "config.yaml"]
    return _file_digests(config.run.benchmark_root, paths)


def _local_evaluation_sources(adapter_name: str) -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "evaluation" / adapter_name,
        root / "evaluation/ibm.py",
    ]
    return _file_digests(root, paths)


def _file_digests(root: Path, paths: list[Path]) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in paths
    }

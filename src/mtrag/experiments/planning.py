from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mtrag.data.benchmark import DOMAINS
from mtrag.experiments.common import ollama_model_info
from mtrag.experiments.fingerprint import file_sha256, fingerprint
from mtrag.experiments.spec import ExperimentConfig, PipelineConfig
from mtrag.llm.history_agent import AGENT_PROTOCOL_VERSION
from mtrag.llm.prompts import (
    GROUNDED_COMPOSITION_VERSION,
    HISTORY_ANSWER_VERSION,
    HISTORY_QUESTION_VERSION,
)
from mtrag.runtime import ResourceRequest, StageSpec
from mtrag.runtime.state import write_json_atomic
from mtrag.schemas import ArtifactRef

ALL_TASK_LAST_QUERY_VERSION = "generation-final-user-v1"
GENERATION_TASKS = "mtrag-human/generation_tasks/reference.jsonl"
TASK_A_SOURCES = ("scripts/evaluation/run_retrieval_eval.py",) + tuple(
    f"mtrag-human/retrieval_tasks/{domain}/qrels/dev.tsv" for domain in DOMAINS
)
GENERATION_EVALUATION_SOURCES = (
    "scripts/evaluation/run_algorithmic.py",
    "scripts/evaluation/config.yaml",
)


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
        return next(stage for stage in self.stages if stage.name == name)

    def save(self, run_dir: Path) -> Path:
        document = {
            "stages": [
                {
                    "name": stage.name,
                    "kind": stage.kind,
                    "fingerprint": stage.fingerprint,
                    "params": _encode_params(stage.params),
                    "dependencies": list(stage.dependencies),
                    "cpu_slots": stage.cpu_slots,
                    "gpu": stage.gpu,
                    "max_attempts": stage.max_attempts,
                }
                for stage in self.stages
            ]
        }
        revision = fingerprint("workflow", document)
        path = run_dir / "plans" / f"{revision}.json"
        if not path.exists():
            write_json_atomic(path, document)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Workflow":
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        stages = []
        for record in document["stages"]:
            stage = dict(record)
            stage["params"] = _decode_params(stage["params"])
            stage["dependencies"] = tuple(stage["dependencies"])
            stages.append(PlannedStage(**stage))
        return cls(tuple(stages))


def _encode_params(value: Any) -> Any:
    if isinstance(value, ArtifactRef):
        return {"$artifact": [value.name, value.revision]}
    if isinstance(value, Mapping):
        return {key: _encode_params(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode_params(item) for item in value]
    return value


def _decode_params(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"$artifact"}:
        return ArtifactRef(*value["$artifact"])
    if isinstance(value, dict):
        return {key: _decode_params(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_params(item) for item in value]
    return value


class PlanBuilder:
    """Expand named config outputs into stages, memoizing shared artifacts."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.stages: list[PlannedStage] = []
        self.producers: dict[ArtifactRef, str] = {}
        self.queries: dict[str, ArtifactRef] = {}
        self.features: dict[str, ArtifactRef] = {}
        self.outputs: dict[str, ArtifactRef] = {}
        self.generations: dict[str, ArtifactRef] = {}
        self.generation_evaluations: dict[str, str] = {}
        self.generation_tasks = config.run.benchmark_root / GENERATION_TASKS

    def build(self, schedule_name: str) -> Workflow:
        schedule = self.config.schedule(schedule_name)
        for reference in schedule.task_a:
            self.task_a(reference)
        for job_name in schedule.generation:
            self.generation(job_name)
        self.finish_generation(schedule_name)
        return Workflow(tuple(self.stages))

    def produce(
        self,
        kind: str,
        artifact: ArtifactRef,
        *,
        params: dict[str, Any] | None = None,
        dependencies: tuple[str, ...] = (),
        logical_name: str | None = None,
        cpu_slots: int = 1,
        gpu: bool = False,
        max_attempts: int = 1,
    ) -> ArtifactRef:
        name = f"{kind}.{logical_name or artifact.name}.{artifact.revision[:12]}"
        stage = PlannedStage(
            name,
            kind,
            artifact.revision,
            params or {},
            dependencies,
            cpu_slots,
            gpu,
            max_attempts,
        )
        self.stages.append(stage)
        self.producers[artifact] = name
        return artifact

    def after(self, *artifacts: ArtifactRef) -> tuple[str, ...]:
        return tuple(
            self.producers[item] for item in artifacts if item in self.producers
        )

    def query(self, name: str) -> ArtifactRef:
        if name in self.queries:
            return self.queries[name]
        query = self.config.query(name)
        inputs = {"kind": query.kind, "sources": self.query_sources(query.kind)}
        if query.kind == "last_turn_all":
            inputs["semantics"] = ALL_TASK_LAST_QUERY_VERSION
        if query.kind in {"rewrite", "agentic"}:
            model = ollama_model_info(self.config)
            inputs.update(
                prompt_sha256=file_sha256(query.prompt),
                temperature=query.temperature,
                max_tokens=query.max_tokens,
                **model.provenance,
                semantics=(
                    AGENT_PROTOCOL_VERSION if query.kind == "agentic"
                    else "ollama-rewrite-v1"
                ),
            )
            if query.kind == "agentic":
                inputs.update(
                    answer_prompt_sha256=file_sha256(query.answer_prompt),
                    compose_prompt_sha256=file_sha256(query.compose_prompt),
                    question_semantics=HISTORY_QUESTION_VERSION,
                    answer_semantics=HISTORY_ANSWER_VERSION,
                    composition_semantics=GROUNDED_COMPOSITION_VERSION,
                )
        artifact = _artifact(name, "query", **inputs)
        if query.kind in {"rewrite", "agentic"}:
            self.produce(
                "rewrite",
                artifact,
                params={"query": artifact},
                gpu=True,
                max_attempts=2,
            )
        self.queries[name] = artifact
        return artifact

    def query_features(self, query_name: str) -> ArtifactRef:
        if query_name in self.features:
            return self.features[query_name]
        query = self.query(query_name)
        artifact = _artifact(
            query_name,
            "bge-query-features",
            "flagembedding-bge-m3-v1",
            query=query.revision,
            model_revision=self.config.models.bge_revision,
            max_length=self.config.models.bge_max_length,
        )
        self.produce(
            "encode",
            artifact,
            params={"query": query, "features": artifact},
            dependencies=self.after(query),
            logical_name=f"bge.{query_name}",
            gpu=True,
        )
        self.features[query_name] = artifact
        return artifact

    def output(self, reference: str) -> ArtifactRef:
        if reference in self.outputs:
            return self.outputs[reference]
        pipeline, output = self.config.resolve_retrieval_output(reference)
        query = self.query(pipeline.query)

        if pipeline.kind == "bge" and output in {"dense", "sparse"}:
            artifact = self.bge_output(reference, pipeline.query, query, output)
        elif pipeline.kind == "bge" and output == "rrf":
            artifact = self.rrf_output(reference, pipeline.name)
        elif pipeline.kind == "elser" and output == "base":
            artifact = self.elser_output(reference, query)
        else:
            artifact = self.reranked_output(reference, pipeline, query)
        self.outputs[reference] = artifact
        return artifact

    def bge_output(
        self, reference: str, query_name: str, query: ArtifactRef, method: str
    ) -> ArtifactRef:
        features = self.query_features(query_name)
        dense = method == "dense"
        retrieval, models = self.config.retrieval, self.config.models
        candidate_multiplier = retrieval.dense_candidate_multiplier if dense else None
        rescore_oversample = retrieval.dense_rescore_oversample if dense else None
        artifact = _artifact(
            reference,
            f"bge-{method}",
            "elasticsearch-retrieval-v1",
            features=features.revision,
            model_revision=models.bge_revision,
            index_revision=retrieval.bge_index_revision,
            top_k=getattr(retrieval, f"{method}_top_k"),
            candidate_multiplier=candidate_multiplier,
            rescore_oversample=rescore_oversample,
        )
        params = {
            "output": artifact,
            "query": query,
            "features": features,
            "method": method,
        }
        return self.produce(
            "retrieve",
            artifact,
            params=params,
            dependencies=self.after(features),
            cpu_slots=max(1, self.config.run.cpu_slots - 1),
            max_attempts=2,
        )

    def rrf_output(self, reference: str, pipeline_name: str) -> ArtifactRef:
        dense = self.output(f"{pipeline_name}.dense")
        sparse = self.output(f"{pipeline_name}.sparse")
        artifact = _artifact(
            reference,
            "rrf",
            "rrf-v1",
            dense=dense.revision,
            sparse=sparse.revision,
            top_k=self.config.retrieval.rrf_top_k,
            rank_constant=self.config.retrieval.rrf_rank_constant,
        )
        return self.produce(
            "fuse",
            artifact,
            params={"output": artifact, "dense": dense, "sparse": sparse},
            dependencies=self.after(dense, sparse),
        )

    def elser_output(self, reference: str, query: ArtifactRef) -> ArtifactRef:
        artifact = _artifact(
            reference,
            "elser",
            "elasticsearch-retrieval-v1",
            query=query.revision,
            inference_id=self.config.services.elser_inference_id,
            index_revision=self.config.retrieval.elser_index_revision,
            top_k=self.config.retrieval.elser_top_k,
        )
        return self.produce(
            "retrieve",
            artifact,
            params={"output": artifact, "query": query, "method": "elser"},
            dependencies=self.after(query),
            cpu_slots=max(1, self.config.run.cpu_slots - 1),
            max_attempts=2,
        )

    def reranked_output(
        self, reference: str, pipeline: PipelineConfig, query: ArtifactRef
    ) -> ArtifactRef:
        source_kind = "rrf" if pipeline.kind == "bge" else "base"
        source = self.output(f"{pipeline.name}.{source_kind}")
        artifact = _artifact(
            reference,
            "rerank",
            "bge-v2-m3-rerank-v1",
            source=source.revision,
            query=query.revision,
            model_revision=self.config.models.reranker_revision,
            max_length=self.config.models.reranker_max_length,
            input_top_k=self.config.reranking.input_top_k,
            output_top_k=self.config.reranking.output_top_k,
        )
        return self.produce(
            "rerank",
            artifact,
            params={"output": artifact, "source": source, "query": query},
            dependencies=self.after(source),
            gpu=True,
        )

    def task_a(self, reference: str) -> None:
        output = self.output(reference)
        evaluation = _artifact(
            reference,
            "ibm-task-a",
            retrieval=output.revision,
            prediction_top_k=self.config.retrieval.prediction_top_k,
            task_source=file_sha256(self.generation_tasks),
            official_sources=_file_digests(
                self.config.run.benchmark_root, TASK_A_SOURCES
            ),
            adapter_sources=_local_evaluation_sources("retrieval.py"),
        )
        self.produce(
            "evaluate_task_a",
            evaluation,
            params={"output": output, "evaluation_revision": evaluation.revision},
            dependencies=self.after(output),
        )

    def generation(self, job_name: str) -> None:
        if job_name in self.generations:
            return
        job = self.config.generation_job(job_name)
        generator = self.config.generator(job.generator)
        task_revision = file_sha256(self.generation_tasks)
        context = None if job.contexts == "reference" else self.output(job.contexts)
        artifact = _artifact(
            job_name,
            "generation",
            "ollama-grounded-generation-v1",
            task=job.task,
            task_source=task_revision,
            contexts=context.revision if context else task_revision,
            **ollama_model_info(self.config).provenance,
            prompt_sha256=file_sha256(generator.prompt),
            temperature=generator.temperature,
            max_tokens=generator.max_tokens,
            context_top_k=generator.context_top_k,
        )
        params = {"generation": artifact}
        if context:
            params["context"] = context
        self.produce(
            "generate",
            artifact,
            params=params,
            dependencies=self.after(context) if context else (),
            gpu=True,
            max_attempts=2,
        )
        self.generations[job_name] = artifact
        if job.evaluate:
            self.generation_evaluations[job_name] = _artifact(
                job_name,
                "ibm-generation-evaluation",
                generation=artifact.revision,
                official_sources=_file_digests(
                    self.config.run.benchmark_root, GENERATION_EVALUATION_SOURCES
                ),
                bertscore_model=self.config.evaluation.bertscore_model,
                adapter_sources=_local_evaluation_sources("generation.py"),
            ).revision

    def finish_generation(self, schedule_name: str) -> None:
        if not self.generations:
            return
        unload = _artifact(
            schedule_name,
            "ollama-unload",
            "ollama-unload-v1",
            generations=sorted(item.revision for item in self.generations.values()),
            model=self.config.models.ollama_model,
        )
        self.produce(
            "unload_ollama",
            unload,
            dependencies=self.after(*self.generations.values()),
            gpu=True,
        )
        jobs = [
            {"generation": self.generations[name], "evaluation_revision": revision}
            for name, revision in self.generation_evaluations.items()
        ]
        if jobs:
            evaluation = _artifact(
                schedule_name,
                "ibm-generation-evaluation-batch",
                evaluations=sorted(self.generation_evaluations.values()),
            )
            self.produce(
                "evaluate_generation_batch",
                evaluation,
                params={"jobs": jobs},
                dependencies=self.after(unload),
                gpu=True,
            )

    def query_sources(self, kind: str) -> dict[str, str]:
        root = self.config.run.benchmark_root
        if kind in {"last_turn_all", "rewrite", "agentic"}:
            return _file_digests(root, [GENERATION_TASKS])
        suffix = "lastturn" if kind == "last_turn" else "rewrite"
        paths = [
            f"mtrag-human/retrieval_tasks/{domain}/{domain}_{suffix}.jsonl"
            for domain in DOMAINS
        ]
        return _file_digests(root, paths)


def build_workflow(config: ExperimentConfig, *, schedule: str) -> Workflow:
    return PlanBuilder(config).build(schedule)


def _artifact(
    name: str, fingerprint_kind: str, semantics: str | None = None, /, **inputs: Any
) -> ArtifactRef:
    if semantics:
        inputs["semantics"] = semantics
    return ArtifactRef(name, fingerprint(fingerprint_kind, inputs))


def build_plan(
    config: ExperimentConfig,
    run_dir: str | Path,
    *,
    workflow: Workflow,
    plan_path: Path,
) -> tuple[StageSpec, ...]:
    root = Path(run_dir).expanduser().resolve()
    script = config.project_root / "scripts" / "run_experiment.py"
    return tuple(
        StageSpec(
            name=stage.name,
            command=(
                sys.executable, str(script), "stage", stage.name,
                "--plan", str(plan_path),
                "--config", str(config.path),
                "--run-dir", str(root),
            ),
            dependencies=stage.dependencies,
            resources=ResourceRequest(stage.cpu_slots, stage.gpu),
            cwd=config.project_root,
            max_attempts=stage.max_attempts,
            retry_delay=5.0,
        )
        for stage in workflow.stages
    )


def _local_evaluation_sources(adapter_name: str) -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return _file_digests(root, [f"evaluation/{adapter_name}", "evaluation/ibm.py"])


def _file_digests(root: Path, paths: Iterable[str]) -> dict[str, str]:
    return {path: file_sha256(root / path) for path in paths}

from __future__ import annotations

import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mtrag.data.benchmark import DOMAINS
from mtrag.experiments.fingerprint import file_sha256, fingerprint
from mtrag.experiments.spec import ExperimentConfig
from mtrag.llm.history_agent import AGENT_PROTOCOL_VERSION
from mtrag.llm.prompts import (
    GROUNDED_COMPOSITION_VERSION,
    HISTORY_ANSWER_VERSION,
    HISTORY_QUESTION_VERSION,
)
from mtrag.runtime import ResourceRequest, StageSpec


ALL_TASK_LAST_QUERY_VERSION = "generation-final-user-v1"


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


@dataclass(frozen=True, slots=True)
class _Product:
    revision: str
    stage: str | None = None


@dataclass
class _Builder:
    config: ExperimentConfig
    stages: dict[str, PlannedStage] = field(default_factory=dict)
    queries: dict[str, _Product] = field(default_factory=dict)
    features: dict[str, _Product] = field(default_factory=dict)
    outputs: dict[str, _Product] = field(default_factory=dict)
    generations: dict[str, _Product] = field(default_factory=dict)
    generation_evaluations: dict[str, str] = field(default_factory=dict)

    def build(self, schedule_name: str) -> Workflow:
        schedule = self.config.schedule(schedule_name)
        for reference in schedule.task_a:
            self._task_a(reference)
        for job_name in schedule.generation:
            self._generation(job_name)
        self._finish_generation(schedule_name)
        return Workflow(stages=tuple(self.stages.values()))

    def _query(self, name: str) -> _Product:
        if name in self.queries:
            return self.queries[name]
        query = self.config.query(name)
        inputs: dict[str, Any] = {
            "kind": query.kind,
            "sources": _query_sources(self.config, query.kind),
        }
        if query.kind == "last_turn_all":
            inputs["semantics"] = ALL_TASK_LAST_QUERY_VERSION
        if query.kind in {"rewrite", "agentic"}:
            inputs.update(
                {
                    "prompt_sha256": file_sha256(query.prompt),
                    "temperature": query.temperature,
                    "max_tokens": query.max_tokens,
                    **_ollama_identity(self.config),
                    "semantics": (
                        AGENT_PROTOCOL_VERSION
                        if query.kind == "agentic"
                        else "ollama-rewrite-v1"
                    ),
                }
            )
            if query.kind == "agentic":
                inputs["answer_prompt_sha256"] = file_sha256(
                    query.answer_prompt
                )
                inputs["compose_prompt_sha256"] = file_sha256(
                    query.compose_prompt
                )
                inputs["question_semantics"] = HISTORY_QUESTION_VERSION
                inputs["answer_semantics"] = HISTORY_ANSWER_VERSION
                inputs["composition_semantics"] = GROUNDED_COMPOSITION_VERSION
        revision = fingerprint("query", inputs)
        if query.kind in {"rewrite", "agentic"}:
            product = self._add(
                "rewrite",
                name,
                revision,
                gpu=True,
                max_attempts=2,
                query_name=name,
                query_revision=revision,
            )
        else:
            product = _Product(revision)
        self.queries[name] = product
        return product

    def _features(self, query_name: str) -> _Product:
        if query_name in self.features:
            return self.features[query_name]
        query = self._query(query_name)
        revision = _revision(
            "bge-query-features",
            "flagembedding-bge-m3-v1",
            query=query.revision,
            model_revision=self.config.models.bge_revision,
            max_length=self.config.models.bge_max_length,
        )
        product = self._add(
            "encode",
            f"bge.{query_name}",
            revision,
            dependencies=_after(query),
            gpu=True,
            query_name=query_name,
            query_revision=query.revision,
            feature_revision=revision,
        )
        self.features[query_name] = product
        return product

    def _output(self, reference: str) -> _Product:
        if reference in self.outputs:
            return self.outputs[reference]
        pipeline, output = self.config.resolve_retrieval_output(reference)
        query = self._query(pipeline.query)

        if pipeline.kind == "bge" and output in {"dense", "sparse"}:
            features = self._features(pipeline.query)
            revision = _revision(
                f"bge-{output}",
                "elasticsearch-retrieval-v1",
                features=features.revision,
                model_revision=self.config.models.bge_revision,
                index_revision=self.config.retrieval.bge_index_revision,
                top_k=getattr(self.config.retrieval, f"{output}_top_k"),
                candidate_multiplier=self.config.retrieval.dense_candidate_multiplier
                if output == "dense"
                else None,
                rescore_oversample=self.config.retrieval.dense_rescore_oversample
                if output == "dense"
                else None,
            )
            product = self._add(
                "retrieve",
                reference,
                revision,
                dependencies=_after(features),
                cpu_slots=max(1, self.config.run.cpu_slots - 1),
                max_attempts=2,
                reference=reference,
                revision=revision,
                query_name=pipeline.query,
                query_revision=query.revision,
                feature_revision=features.revision,
                method=output,
            )
        elif pipeline.kind == "bge" and output == "rrf":
            dense_reference = f"{pipeline.name}.dense"
            sparse_reference = f"{pipeline.name}.sparse"
            dense = self._output(dense_reference)
            sparse = self._output(sparse_reference)
            revision = _revision(
                "rrf",
                "rrf-v1",
                dense=dense.revision,
                sparse=sparse.revision,
                top_k=self.config.retrieval.rrf_top_k,
                rank_constant=self.config.retrieval.rrf_rank_constant,
            )
            product = self._add(
                "fuse",
                reference,
                revision,
                dependencies=_after(dense, sparse),
                reference=reference,
                revision=revision,
                dense_reference=dense_reference,
                dense_revision=dense.revision,
                sparse_reference=sparse_reference,
                sparse_revision=sparse.revision,
            )
        elif pipeline.kind == "elser" and output == "base":
            revision = _revision(
                "elser",
                "elasticsearch-retrieval-v1",
                query=query.revision,
                inference_id=self.config.services.elser_inference_id,
                index_revision=self.config.retrieval.elser_index_revision,
                top_k=self.config.retrieval.elser_top_k,
            )
            product = self._add(
                "retrieve",
                reference,
                revision,
                dependencies=_after(query),
                cpu_slots=max(1, self.config.run.cpu_slots - 1),
                max_attempts=2,
                reference=reference,
                revision=revision,
                query_name=pipeline.query,
                query_revision=query.revision,
                method="elser",
            )
        else:
            source_output = "rrf" if pipeline.kind == "bge" else "base"
            source_reference = f"{pipeline.name}.{source_output}"
            source = self._output(source_reference)
            revision = _revision(
                "rerank",
                "bge-v2-m3-rerank-v1",
                source=source.revision,
                query=query.revision,
                model_revision=self.config.models.reranker_revision,
                max_length=self.config.models.reranker_max_length,
                input_top_k=self.config.reranking.input_top_k,
                output_top_k=self.config.reranking.output_top_k,
            )
            product = self._add(
                "rerank",
                reference,
                revision,
                dependencies=_after(source),
                gpu=True,
                reference=reference,
                revision=revision,
                source_reference=source_reference,
                source_revision=source.revision,
                query_name=pipeline.query,
                query_revision=query.revision,
            )

        self.outputs[reference] = product
        return product

    def _task_a(self, reference: str) -> None:
        output = self._output(reference)
        revision = _revision(
            "ibm-task-a",
            retrieval=output.revision,
            prediction_top_k=self.config.retrieval.prediction_top_k,
            task_source=file_sha256(_generation_tasks(self.config)),
            official_sources=_task_a_sources(self.config),
            adapter_sources=_local_evaluation_sources("retrieval.py"),
        )
        self._add(
            "evaluate_task_a",
            reference,
            revision,
            dependencies=_after(output),
            reference=reference,
            revision=output.revision,
            evaluation_revision=revision,
        )

    def _generation(self, job_name: str) -> None:
        if job_name in self.generations:
            return
        job = self.config.generation_job(job_name)
        generator = self.config.generator(job.generator)
        params: dict[str, Any] = {"job_name": job_name}
        task_revision = file_sha256(_generation_tasks(self.config))
        if job.contexts == "reference":
            context = _Product(task_revision)
        else:
            context = self._output(job.contexts)
            params.update(
                {
                    "context_reference": job.contexts,
                    "context_revision": context.revision,
                }
            )
        revision = _revision(
            "generation",
            "ollama-grounded-generation-v1",
            task=job.task,
            task_source=task_revision,
            contexts=context.revision,
            **_ollama_identity(self.config),
            prompt_sha256=file_sha256(generator.prompt),
            temperature=generator.temperature,
            max_tokens=generator.max_tokens,
            context_top_k=generator.context_top_k,
        )
        params["revision"] = revision
        product = self._add(
            "generate",
            job_name,
            revision,
            dependencies=_after(context),
            gpu=True,
            max_attempts=2,
            **params,
        )
        self.generations[job_name] = product
        if not job.evaluate:
            return
        evaluation_revision = _revision(
            "ibm-generation-evaluation",
            generation=revision,
            official_sources=_generation_evaluation_sources(self.config),
            bertscore_model=self.config.evaluation.bertscore_model,
            adapter_sources=_local_evaluation_sources("generation.py"),
        )
        self.generation_evaluations[job_name] = evaluation_revision

    def _finish_generation(self, schedule_name: str) -> None:
        if not self.generations:
            return

        generation_stages = _after(*self.generations.values())
        unload_revision = _revision(
            "ollama-unload",
            "ollama-unload-v1",
            generations=sorted(
                product.revision for product in self.generations.values()
            ),
            model=self.config.models.ollama_model,
        )
        unload_stage = self._add(
            "unload_ollama",
            schedule_name,
            unload_revision,
            dependencies=generation_stages,
            gpu=True,
        )

        jobs = [
            {
                "job_name": job_name,
                "generation_revision": self.generations[job_name].revision,
                "evaluation_revision": evaluation_revision,
            }
            for job_name, evaluation_revision in self.generation_evaluations.items()
        ]
        if not jobs:
            return
        batch_revision = _revision(
            "ibm-generation-evaluation-batch",
            evaluations=sorted(self.generation_evaluations.values()),
        )
        self._add(
            "evaluate_generation_batch",
            schedule_name,
            batch_revision,
            dependencies=_after(unload_stage),
            gpu=True,
            jobs=jobs,
        )

    def _add(
        self,
        kind: str,
        logical_name: str,
        fingerprint_value: str,
        *,
        dependencies: tuple[str, ...] = (),
        cpu_slots: int = 1,
        gpu: bool = False,
        max_attempts: int = 1,
        **params: Any,
    ) -> _Product:
        prefix = f"{kind}.{logical_name}" if logical_name else kind
        name = f"{prefix}.{fingerprint_value[:12]}"
        stage = PlannedStage(
            name=name,
            kind=kind,
            fingerprint=fingerprint_value,
            params=params,
            dependencies=dependencies,
            cpu_slots=cpu_slots,
            gpu=gpu,
            max_attempts=max_attempts,
        )
        self.stages[name] = stage
        return _Product(fingerprint_value, name)


def build_workflow(config: ExperimentConfig, *, schedule: str) -> Workflow:
    return _Builder(config).build(schedule)


def _after(*products: _Product) -> tuple[str, ...]:
    return tuple(product.stage for product in products if product.stage is not None)


def _revision(kind: str, semantics: str | None = None, **inputs: Any) -> str:
    if semantics:
        inputs["semantics"] = semantics
    return fingerprint(kind, inputs)


def _ollama_identity(config: ExperimentConfig) -> dict[str, Any]:
    model = config.models
    return {
        "model": model.ollama_model,
        "model_digest": model.ollama_digest,
        "num_ctx": model.ollama_num_ctx,
        "seed": model.ollama_seed,
    }


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
    return config.run.benchmark_root / "mtrag-human/generation_tasks/reference.jsonl"


def _query_sources(config: ExperimentConfig, kind: str) -> dict[str, str]:
    root = config.run.benchmark_root
    if kind in {"last_turn_all", "rewrite", "agentic"}:
        return _file_digests(root, ["mtrag-human/generation_tasks/reference.jsonl"])
    suffix = "lastturn" if kind == "last_turn" else "rewrite"
    paths = [
        f"mtrag-human/retrieval_tasks/{domain}/{domain}_{suffix}.jsonl"
        for domain in DOMAINS
    ]
    return _file_digests(root, paths)


def _task_a_sources(config: ExperimentConfig) -> dict[str, str]:
    root = config.run.benchmark_root
    paths = ["scripts/evaluation/run_retrieval_eval.py"] + [
        f"mtrag-human/retrieval_tasks/{domain}/qrels/dev.tsv"
        for domain in DOMAINS
    ]
    return _file_digests(root, paths)


def _generation_evaluation_sources(config: ExperimentConfig) -> dict[str, str]:
    return _file_digests(
        config.run.benchmark_root,
        ["scripts/evaluation/run_algorithmic.py", "scripts/evaluation/config.yaml"],
    )


def _local_evaluation_sources(adapter_name: str) -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return _file_digests(root, [f"evaluation/{adapter_name}", "evaluation/ibm.py"])


def _file_digests(root: Path, paths: Iterable[str]) -> dict[str, str]:
    return {path: file_sha256(root / path) for path in paths}

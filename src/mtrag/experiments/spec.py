from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


def _section(document: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = document.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"TOML section [{name}] must be a table")
    return value


def _path(value: str, project_root: Path) -> Path:
    expanded = Path(os.path.expandvars(value)).expanduser()
    return expanded if expanded.is_absolute() else project_root / expanded


def _required_string(section: Mapping[str, Any], name: str, label: str) -> str:
    value = section.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}.{name} must be a non-empty string")
    return value.strip()


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be an array of strings")
    result = tuple(value)
    if any(not isinstance(item, str) or not item.strip() for item in result):
        raise ValueError(f"{label} must be an array of non-empty strings")
    return tuple(item.strip() for item in result)


_SAFE_NAME = re.compile(r"[a-z][a-z0-9_]*")
PIPELINE_OUTPUTS = {
    "bge": frozenset(("dense", "sparse", "rrf", "rrf_reranked")),
    "elser": frozenset(("base", "reranked")),
}


@dataclass(frozen=True, slots=True)
class RunConfig:
    name: str
    output_root: Path
    benchmark_root: Path
    cpu_slots: int


@dataclass(frozen=True, slots=True)
class ServiceConfig:
    elasticsearch_url: str
    elser_inference_id: str
    ollama_url: str


@dataclass(frozen=True, slots=True)
class ModelConfig:
    bge_path: Path
    bge_revision: str
    bge_batch_size: int
    bge_max_length: int
    reranker_path: Path
    reranker_revision: str
    reranker_batch_size: int
    reranker_max_length: int
    ollama_model: str
    ollama_digest: str
    ollama_num_ctx: int
    ollama_keep_alive: str
    ollama_seed: int
    ollama_timeout: int


@dataclass(frozen=True, slots=True)
class QueryConfig:
    name: str
    kind: str
    prompt: Path | None
    temperature: float | None
    max_tokens: int | None


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    name: str
    kind: str
    query: str


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    name: str
    prompt: Path
    temperature: float
    max_tokens: int
    context_top_k: int


@dataclass(frozen=True, slots=True)
class GenerationJobConfig:
    name: str
    task: str
    generator: str
    contexts: str
    evaluate: bool


@dataclass(frozen=True, slots=True)
class ScheduleConfig:
    name: str
    task_a: tuple[str, ...]
    generation: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    bge_index_revision: str
    elser_index_revision: str
    dense_top_k: int
    dense_candidate_multiplier: int
    dense_rescore_oversample: float
    sparse_top_k: int
    elser_top_k: int
    rrf_top_k: int
    rrf_rank_constant: int
    prediction_top_k: int
    request_batch_size: int


@dataclass(frozen=True, slots=True)
class RerankingConfig:
    input_top_k: int
    output_top_k: int
    task_batch_size: int


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    bertscore_model: str
    bertscore_batch_size: int


@dataclass(frozen=True, slots=True)
class ThermalConfig:
    gpu_pause: float
    gpu_resume: float
    cpu_pause: float
    cpu_resume: float
    poll_interval: float
    resume_hold: float


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    path: Path
    project_root: Path
    run: RunConfig
    services: ServiceConfig
    models: ModelConfig
    queries: tuple[QueryConfig, ...]
    pipelines: tuple[PipelineConfig, ...]
    generators: tuple[GeneratorConfig, ...]
    generation_jobs: tuple[GenerationJobConfig, ...]
    schedules: tuple[ScheduleConfig, ...]
    retrieval: RetrievalConfig
    reranking: RerankingConfig
    evaluation: EvaluationConfig
    thermal: ThermalConfig

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        config_path = Path(path).expanduser().resolve()
        document = tomllib.loads(config_path.read_text(encoding="utf-8"))
        project_root = config_path.parent.parent

        run = _section(document, "run")
        services = _section(document, "services")
        models = _section(document, "models")
        queries = _section(document, "queries")
        pipelines = _section(document, "pipelines")
        generators = _section(document, "generators")
        generation_jobs = _section(document, "generation")
        schedules = _section(document, "schedules")
        retrieval = _section(document, "retrieval")
        reranking = _section(document, "reranking")
        evaluation = _section(document, "evaluation")
        thermal = _section(document, "thermal")

        config = cls(
            path=config_path,
            project_root=project_root,
            run=RunConfig(
                name=str(run.get("name", "main")),
                output_root=_path(str(run.get("output_root", "runs")), project_root),
                benchmark_root=_path(
                    str(run.get("benchmark_root", "../mt-rag-benchmark")),
                    project_root,
                ),
                cpu_slots=int(run.get("cpu_slots", 4)),
            ),
            services=ServiceConfig(
                elasticsearch_url=str(
                    services.get("elasticsearch_url", "http://127.0.0.1:9200")
                ).rstrip("/"),
                elser_inference_id=str(
                    services.get("elser_inference_id", "mtrag-elser")
                ),
                ollama_url=str(
                    services.get("ollama_url", "http://127.0.0.1:11434")
                ).rstrip("/"),
            ),
            models=ModelConfig(
                bge_path=_path(
                    str(models.get("bge_path", "~/.cache/mtrag/models/bge-m3")),
                    project_root,
                ),
                bge_revision=str(models.get("bge_revision", "local")),
                bge_batch_size=int(models.get("bge_batch_size", 32)),
                bge_max_length=int(models.get("bge_max_length", 512)),
                reranker_path=_path(
                    str(
                        models.get(
                            "reranker_path",
                            "~/.cache/mtrag/models/bge-reranker-v2-m3",
                        )
                    ),
                    project_root,
                ),
                reranker_revision=str(models.get("reranker_revision", "local")),
                reranker_batch_size=int(models.get("reranker_batch_size", 8)),
                reranker_max_length=int(models.get("reranker_max_length", 512)),
                ollama_model=str(models.get("ollama_model", "qwen3.5:4b-q4_K_M")),
                ollama_digest=str(models.get("ollama_digest", "")),
                ollama_num_ctx=int(models.get("ollama_num_ctx", 8192)),
                ollama_keep_alive=str(models.get("ollama_keep_alive", "10m")),
                ollama_seed=int(models.get("ollama_seed", 42)),
                ollama_timeout=int(models.get("ollama_timeout", 600)),
            ),
            queries=tuple(
                _query_config(str(name), _section(queries, str(name)), project_root)
                for name in queries
            ),
            pipelines=tuple(
                PipelineConfig(
                    name=str(name),
                    kind=_required_string(
                        _section(pipelines, str(name)),
                        "kind",
                        f"pipelines.{name}",
                    ),
                    query=_required_string(
                        _section(pipelines, str(name)),
                        "query",
                        f"pipelines.{name}",
                    ),
                )
                for name in pipelines
            ),
            generators=tuple(
                _generator_config(
                    str(name),
                    _section(generators, str(name)),
                    project_root,
                )
                for name in generators
            ),
            generation_jobs=tuple(
                _generation_job_config(
                    str(name),
                    _section(generation_jobs, str(name)),
                )
                for name in generation_jobs
            ),
            schedules=tuple(
                ScheduleConfig(
                    name=str(name),
                    task_a=_string_tuple(
                        _section(schedules, str(name)).get("task_a"),
                        f"schedules.{name}.task_a",
                    ),
                    generation=_string_tuple(
                        _section(schedules, str(name)).get("generation"),
                        f"schedules.{name}.generation",
                    ),
                )
                for name in schedules
            ),
            retrieval=RetrievalConfig(
                bge_index_revision=_required_string(
                    retrieval,
                    "bge_index_revision",
                    "retrieval",
                ),
                elser_index_revision=_required_string(
                    retrieval,
                    "elser_index_revision",
                    "retrieval",
                ),
                dense_top_k=int(retrieval.get("dense_top_k", 50)),
                dense_candidate_multiplier=int(
                    retrieval.get("dense_candidate_multiplier", 10)
                ),
                dense_rescore_oversample=float(
                    retrieval.get("dense_rescore_oversample", 2.0)
                ),
                sparse_top_k=int(retrieval.get("sparse_top_k", 50)),
                elser_top_k=int(retrieval.get("elser_top_k", 20)),
                rrf_top_k=int(retrieval.get("rrf_top_k", 20)),
                rrf_rank_constant=int(retrieval.get("rrf_rank_constant", 60)),
                prediction_top_k=int(retrieval.get("prediction_top_k", 10)),
                request_batch_size=int(retrieval.get("request_batch_size", 64)),
            ),
            reranking=RerankingConfig(
                input_top_k=int(reranking.get("input_top_k", 20)),
                output_top_k=int(reranking.get("output_top_k", 10)),
                task_batch_size=int(reranking.get("task_batch_size", 32)),
            ),
            evaluation=EvaluationConfig(
                bertscore_model=str(
                    evaluation.get(
                        "bertscore_model",
                        "microsoft/deberta-xlarge-mnli",
                    )
                ),
                bertscore_batch_size=int(
                    evaluation.get("bertscore_batch_size", 2)
                ),
            ),
            thermal=ThermalConfig(
                gpu_pause=float(thermal.get("gpu_pause", 80.0)),
                gpu_resume=float(thermal.get("gpu_resume", 72.0)),
                cpu_pause=float(thermal.get("cpu_pause", 90.0)),
                cpu_resume=float(thermal.get("cpu_resume", 80.0)),
                poll_interval=float(thermal.get("poll_interval", 5.0)),
                resume_hold=float(thermal.get("resume_hold", 30.0)),
            ),
        )
        config.validate()
        return config

    @property
    def default_run_dir(self) -> Path:
        return self.run.output_root / self.run.name

    def query(self, name: str) -> QueryConfig:
        for query in self.queries:
            if query.name == name:
                return query
        raise ValueError(f"unknown query: {name}")

    def pipeline(self, name: str) -> PipelineConfig:
        for pipeline in self.pipelines:
            if pipeline.name == name:
                return pipeline
        raise ValueError(f"unknown pipeline: {name}")

    def generator(self, name: str) -> GeneratorConfig:
        for generator in self.generators:
            if generator.name == name:
                return generator
        raise ValueError(f"unknown generator: {name}")

    def generation_job(self, name: str) -> GenerationJobConfig:
        for job in self.generation_jobs:
            if job.name == name:
                return job
        raise ValueError(f"unknown generation job: {name}")

    def schedule(self, name: str) -> ScheduleConfig:
        for schedule in self.schedules:
            if schedule.name == name:
                return schedule
        raise ValueError(f"unknown schedule: {name}")

    def resolve_retrieval_output(self, reference: str) -> tuple[PipelineConfig, str]:
        pipeline_name, separator, output = reference.partition(".")
        if not separator or "." in output:
            raise ValueError(
                f"retrieval output must be '<pipeline>.<output>': {reference!r}"
            )
        pipeline = self.pipeline(pipeline_name)
        if output not in PIPELINE_OUTPUTS[pipeline.kind]:
            raise ValueError(
                f"pipeline {pipeline.name!r} ({pipeline.kind}) has no output "
                f"{output!r}"
            )
        return pipeline, output

    def validate(self) -> None:
        positive = {
            "run.cpu_slots": self.run.cpu_slots,
            "models.bge_batch_size": self.models.bge_batch_size,
            "models.bge_max_length": self.models.bge_max_length,
            "models.reranker_batch_size": self.models.reranker_batch_size,
            "models.reranker_max_length": self.models.reranker_max_length,
            "models.ollama_num_ctx": self.models.ollama_num_ctx,
            "models.ollama_timeout": self.models.ollama_timeout,
            "retrieval.dense_top_k": self.retrieval.dense_top_k,
            "retrieval.dense_candidate_multiplier": (
                self.retrieval.dense_candidate_multiplier
            ),
            "retrieval.sparse_top_k": self.retrieval.sparse_top_k,
            "retrieval.elser_top_k": self.retrieval.elser_top_k,
            "retrieval.rrf_top_k": self.retrieval.rrf_top_k,
            "retrieval.rrf_rank_constant": self.retrieval.rrf_rank_constant,
            "retrieval.prediction_top_k": self.retrieval.prediction_top_k,
            "retrieval.request_batch_size": self.retrieval.request_batch_size,
            "reranking.input_top_k": self.reranking.input_top_k,
            "reranking.output_top_k": self.reranking.output_top_k,
            "reranking.task_batch_size": self.reranking.task_batch_size,
            "evaluation.bertscore_batch_size": (
                self.evaluation.bertscore_batch_size
            ),
        }
        positive.update(
            {
                f"queries.{query.name}.max_tokens": query.max_tokens
                for query in self.queries
                if query.max_tokens is not None
            }
        )
        positive.update(
            {
                f"generators.{generator.name}.max_tokens": generator.max_tokens
                for generator in self.generators
            }
        )
        positive.update(
            {
                f"generators.{generator.name}.context_top_k": (
                    generator.context_top_k
                )
                for generator in self.generators
            }
        )
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"configuration values must be positive: {', '.join(invalid)}")
        if self.retrieval.prediction_top_k > 10:
            raise ValueError("retrieval.prediction_top_k cannot exceed the official limit 10")
        if self.retrieval.dense_rescore_oversample < 1.0:
            raise ValueError("dense_rescore_oversample must be at least 1.0")
        if self.evaluation.bertscore_model != "microsoft/deberta-xlarge-mnli":
            raise ValueError(
                "the official IBM evaluator requires "
                "evaluation.bertscore_model='microsoft/deberta-xlarge-mnli'"
            )
        names = (
            *(query.name for query in self.queries),
            *(pipeline.name for pipeline in self.pipelines),
            *(generator.name for generator in self.generators),
            *(job.name for job in self.generation_jobs),
            *(schedule.name for schedule in self.schedules),
        )
        invalid_names = [name for name in names if _SAFE_NAME.fullmatch(name) is None]
        if invalid_names:
            raise ValueError(
                "configuration names must be safe identifiers: "
                + ", ".join(invalid_names)
            )

        invalid_query_kinds = [
            query.name
            for query in self.queries
            if query.kind not in {"last_turn", "gold", "rewrite"}
        ]
        if invalid_query_kinds:
            raise ValueError(
                "unknown query kind: " + ", ".join(invalid_query_kinds)
            )
        invalid_pipeline_kinds = [
            pipeline.name
            for pipeline in self.pipelines
            if pipeline.kind not in PIPELINE_OUTPUTS
        ]
        if invalid_pipeline_kinds:
            raise ValueError(
                "unknown pipeline kind: " + ", ".join(invalid_pipeline_kinds)
            )

        invalid_temperatures = [
            query.name
            for query in self.queries
            if query.temperature is not None
            and not 0.0 <= query.temperature <= 2.0
        ] + [
            generator.name
            for generator in self.generators
            if not 0.0 <= generator.temperature <= 2.0
        ]
        if invalid_temperatures:
            raise ValueError(
                "temperatures must be between 0 and 2: "
                + ", ".join(invalid_temperatures)
            )
        missing_prompts = [
            str(query.prompt)
            for query in self.queries
            if query.prompt is not None and not query.prompt.is_file()
        ] + [
            str(generator.prompt)
            for generator in self.generators
            if not generator.prompt.is_file()
        ]
        if missing_prompts:
            raise ValueError("prompt file is missing: " + ", ".join(missing_prompts))

        for pipeline in self.pipelines:
            self.query(pipeline.query)
        for job in self.generation_jobs:
            self.generator(job.generator)
            if job.task == "b":
                if job.contexts != "reference":
                    raise ValueError(
                        f"generation.{job.name}: Task B contexts must be 'reference'"
                    )
            elif job.task == "c":
                if job.contexts == "reference":
                    raise ValueError(
                        f"generation.{job.name}: Task C requires a retrieval output"
                    )
                self.resolve_retrieval_output(job.contexts)
            else:
                raise ValueError(
                    f"generation.{job.name}.task must be 'b' or 'c'"
                )

        for schedule in self.schedules:
            for reference in schedule.task_a:
                self.resolve_retrieval_output(reference)
            for job_name in schedule.generation:
                self.generation_job(job_name)

        if self.reranking.input_top_k > self.retrieval.rrf_top_k:
            raise ValueError("reranking.input_top_k cannot exceed retrieval.rrf_top_k")
        if self.reranking.output_top_k > self.reranking.input_top_k:
            raise ValueError("reranking.output_top_k cannot exceed reranking.input_top_k")
        oversized_contexts = [
            generator.name
            for generator in self.generators
            if generator.context_top_k > 10
        ]
        if oversized_contexts:
            raise ValueError(
                "generator context_top_k cannot exceed the official limit 10: "
                + ", ".join(oversized_contexts)
            )


def _query_config(
    name: str,
    section: Mapping[str, Any],
    project_root: Path,
) -> QueryConfig:
    kind = _required_string(section, "kind", f"queries.{name}")
    if kind != "rewrite":
        return QueryConfig(
            name=name,
            kind=kind,
            prompt=None,
            temperature=None,
            max_tokens=None,
        )
    return QueryConfig(
        name=name,
        kind=kind,
        prompt=_path(
            _required_string(section, "prompt", f"queries.{name}"),
            project_root,
        ),
        temperature=float(section.get("temperature", 0.0)),
        max_tokens=int(section.get("max_tokens", 128)),
    )


def _generator_config(
    name: str,
    section: Mapping[str, Any],
    project_root: Path,
) -> GeneratorConfig:
    return GeneratorConfig(
        name=name,
        prompt=_path(
            _required_string(section, "prompt", f"generators.{name}"),
            project_root,
        ),
        temperature=float(section.get("temperature", 0.0)),
        max_tokens=int(section.get("max_tokens", 512)),
        context_top_k=int(section.get("context_top_k", 5)),
    )


def _generation_job_config(
    name: str,
    section: Mapping[str, Any],
) -> GenerationJobConfig:
    evaluate = section.get("evaluate", True)
    if not isinstance(evaluate, bool):
        raise ValueError(f"generation.{name}.evaluate must be a boolean")
    return GenerationJobConfig(
        name=name,
        task=_required_string(section, "task", f"generation.{name}"),
        generator=_required_string(
            section,
            "generator",
            f"generation.{name}",
        ),
        contexts=_required_string(section, "contexts", f"generation.{name}"),
        evaluate=evaluate,
    )

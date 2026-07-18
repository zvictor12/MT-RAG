from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


_SAFE_NAME = re.compile(r"[a-z][a-z0-9_]*")

PIPELINE_OUTPUTS = {
    "bge": {"dense", "sparse", "rrf", "rrf_reranked"},
    "elser": {"base", "reranked"},
}
QUERY_KINDS = {"last_turn", "last_turn_all", "gold", "rewrite", "agentic"}


def _path(value: str, project_root: Path) -> Path:
    path = Path(os.path.expandvars(value)).expanduser()
    return path if path.is_absolute() else project_root / path


def _named_tables(
    document: Mapping[str, Any], section: str
) -> Iterator[tuple[str, Mapping[str, Any]]]:
    tables = document.get(section, {})
    if not isinstance(tables, Mapping):
        raise ValueError(f"TOML section [{section}] must be a table")
    for name, table in tables.items():
        if not isinstance(table, Mapping):
            raise ValueError(f"TOML section [{section}.{name}] must be a table")
        yield name, table


@dataclass(frozen=True, slots=True)
class RunConfig:
    name: str = "main"
    output_root: Path = Path("runs")
    benchmark_root: Path = Path("../mt-rag-benchmark")
    cpu_slots: int = 4


@dataclass(frozen=True, slots=True)
class ServiceConfig:
    elasticsearch_url: str = "http://127.0.0.1:9200"
    elser_inference_id: str = "mtrag-elser"
    ollama_url: str = "http://127.0.0.1:11434"


@dataclass(frozen=True, slots=True)
class ModelConfig:
    bge_path: Path = Path("~/.cache/mtrag/models/bge-m3")
    bge_revision: str = "local"
    bge_batch_size: int = 32
    bge_max_length: int = 512
    reranker_path: Path = Path("~/.cache/mtrag/models/bge-reranker-v2-m3")
    reranker_revision: str = "local"
    reranker_batch_size: int = 8
    reranker_max_length: int = 512
    ollama_model: str = "qwen3.5:4b-q4_K_M"
    ollama_digest: str = ""
    ollama_num_ctx: int = 8192
    ollama_keep_alive: str = "10m"
    ollama_seed: int = 42
    ollama_timeout: int = 600


@dataclass(frozen=True, slots=True)
class QueryConfig:
    name: str
    kind: str
    prompt: Path | None = None
    answer_prompt: Path | None = None
    compose_prompt: Path | None = None
    temperature: float | None = None
    max_tokens: int | None = None


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
    bge_index_revision: str = ""
    elser_index_revision: str = ""
    dense_top_k: int = 50
    dense_candidate_multiplier: int = 10
    dense_rescore_oversample: float = 2.0
    sparse_top_k: int = 50
    elser_top_k: int = 20
    rrf_top_k: int = 20
    rrf_rank_constant: int = 60
    prediction_top_k: int = 10
    request_batch_size: int = 64


@dataclass(frozen=True, slots=True)
class RerankingConfig:
    input_top_k: int = 20
    output_top_k: int = 10
    task_batch_size: int = 32


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    bertscore_model: str = "microsoft/deberta-xlarge-mnli"
    bertscore_batch_size: int = 2


@dataclass(frozen=True, slots=True)
class ThermalConfig:
    gpu_pause: float = 80.0
    gpu_resume: float = 72.0
    cpu_pause: float = 90.0
    cpu_resume: float = 80.0
    poll_interval: float = 5.0
    resume_hold: float = 30.0


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    path: Path
    project_root: Path
    run: RunConfig
    services: ServiceConfig
    models: ModelConfig
    queries: dict[str, QueryConfig]
    pipelines: dict[str, PipelineConfig]
    generators: dict[str, GeneratorConfig]
    generation_jobs: dict[str, GenerationJobConfig]
    schedules: dict[str, ScheduleConfig]
    retrieval: RetrievalConfig
    reranking: RerankingConfig
    evaluation: EvaluationConfig
    thermal: ThermalConfig

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        config_path = Path(path).expanduser().resolve()
        project_root = config_path.parent.parent
        document = tomllib.loads(config_path.read_text(encoding="utf-8"))

        sections: dict[str, dict[str, Any]] = {}
        for name in (
            "run", "services", "models", "retrieval", "reranking",
            "evaluation", "thermal",
        ):
            table = document.get(name, {})
            if not isinstance(table, Mapping):
                raise ValueError(f"TOML section [{name}] must be a table")
            sections[name] = dict(table)

        run = sections["run"]
        run_defaults = RunConfig()
        run["output_root"] = _path(
            str(run.get("output_root", run_defaults.output_root)), project_root
        )
        run["benchmark_root"] = _path(
            str(run.get("benchmark_root", run_defaults.benchmark_root)), project_root
        )

        services = sections["services"]
        service_defaults = ServiceConfig()
        for name in ("elasticsearch_url", "ollama_url"):
            services[name] = str(
                services.get(name, getattr(service_defaults, name))
            ).rstrip("/")

        models = sections["models"]
        model_defaults = ModelConfig()
        for name in ("bge_path", "reranker_path"):
            models[name] = _path(
                str(models.get(name, getattr(model_defaults, name))), project_root
            )

        queries: dict[str, QueryConfig] = {}
        for name, table in _named_tables(document, "queries"):
            kind = table.get("kind", "")
            if kind not in {"rewrite", "agentic"}:
                queries[name] = QueryConfig(name=name, kind=str(kind))
                continue
            queries[name] = QueryConfig(
                name=name,
                kind=str(kind),
                prompt=_path(str(table.get("prompt", "")), project_root),
                answer_prompt=(
                    _path(str(table.get("answer_prompt", "")), project_root)
                    if kind == "agentic"
                    else None
                ),
                compose_prompt=(
                    _path(str(table.get("compose_prompt", "")), project_root)
                    if kind == "agentic"
                    else None
                ),
                temperature=float(table.get("temperature", 0.0)),
                max_tokens=int(table.get("max_tokens", 128)),
            )

        pipelines = {
            name: PipelineConfig(
                name=name,
                kind=str(table.get("kind", "")),
                query=str(table.get("query", "")),
            )
            for name, table in _named_tables(document, "pipelines")
        }
        generators = {
            name: GeneratorConfig(
                name=name,
                prompt=_path(str(table.get("prompt", "")), project_root),
                temperature=float(table.get("temperature", 0.0)),
                max_tokens=int(table.get("max_tokens", 512)),
                context_top_k=int(table.get("context_top_k", 5)),
            )
            for name, table in _named_tables(document, "generators")
        }
        generation_jobs = {
            name: GenerationJobConfig(
                name=name,
                task=str(table.get("task", "")),
                generator=str(table.get("generator", "")),
                contexts=str(table.get("contexts", "")),
                evaluate=cast(bool, table.get("evaluate", True)),
            )
            for name, table in _named_tables(document, "generation")
        }
        schedules = {
            name: ScheduleConfig(
                name=name,
                task_a=tuple(cast(list[str], table.get("task_a", []))),
                generation=tuple(cast(list[str], table.get("generation", []))),
            )
            for name, table in _named_tables(document, "schedules")
        }

        config = cls(
            path=config_path,
            project_root=project_root,
            run=RunConfig(**run),
            services=ServiceConfig(**services),
            models=ModelConfig(**models),
            queries=queries,
            pipelines=pipelines,
            generators=generators,
            generation_jobs=generation_jobs,
            schedules=schedules,
            retrieval=RetrievalConfig(**sections["retrieval"]),
            reranking=RerankingConfig(**sections["reranking"]),
            evaluation=EvaluationConfig(**sections["evaluation"]),
            thermal=ThermalConfig(**sections["thermal"]),
        )
        config.validate()
        return config

    @property
    def default_run_dir(self) -> Path:
        return self.run.output_root / self.run.name

    def query(self, name: str) -> QueryConfig:
        return self.queries[name]

    def pipeline(self, name: str) -> PipelineConfig:
        return self.pipelines[name]

    def generator(self, name: str) -> GeneratorConfig:
        return self.generators[name]

    def generation_job(self, name: str) -> GenerationJobConfig:
        return self.generation_jobs[name]

    def schedule(self, name: str) -> ScheduleConfig:
        return self.schedules[name]

    def resolve_retrieval_output(self, reference: str) -> tuple[PipelineConfig, str]:
        pipeline_name, output = reference.split(".", 1)
        return self.pipelines[pipeline_name], output

    def validate(self) -> None:
        errors: list[str] = []
        retrieval, reranking = self.retrieval, self.reranking
        constraints = (
            (
                bool(retrieval.bge_index_revision.strip()),
                "retrieval.bge_index_revision must be a non-empty string",
            ),
            (
                bool(retrieval.elser_index_revision.strip()),
                "retrieval.elser_index_revision must be a non-empty string",
            ),
            (
                1 <= retrieval.prediction_top_k <= 10,
                "retrieval.prediction_top_k cannot exceed the official limit 10",
            ),
            (
                retrieval.dense_rescore_oversample >= 1.0,
                "dense_rescore_oversample must be at least 1.0",
            ),
            (
                reranking.input_top_k <= retrieval.rrf_top_k,
                "reranking.input_top_k cannot exceed retrieval.rrf_top_k",
            ),
            (
                reranking.output_top_k <= reranking.input_top_k,
                "reranking.output_top_k cannot exceed reranking.input_top_k",
            ),
            (
                self.evaluation.bertscore_model == "microsoft/deberta-xlarge-mnli",
                "the official IBM evaluator requires "
                "evaluation.bertscore_model='microsoft/deberta-xlarge-mnli'",
            ),
            (
                self.thermal.gpu_resume < self.thermal.gpu_pause
                and self.thermal.cpu_resume < self.thermal.cpu_pause
                and self.thermal.poll_interval > 0
                and self.thermal.resume_hold >= 0,
                "thermal thresholds must resume below pause and use non-negative timing",
            ),
        )
        errors.extend(message for valid, message in constraints if not valid)

        for section, items in {
            "queries": self.queries,
            "pipelines": self.pipelines,
            "generators": self.generators,
            "generation": self.generation_jobs,
            "schedules": self.schedules,
        }.items():
            for name in items:
                if _SAFE_NAME.fullmatch(name) is None:
                    errors.append(f"{section}.{name} is not a safe identifier")

        for query in self.queries.values():
            label = f"queries.{query.name}"
            if query.kind not in QUERY_KINDS:
                errors.append(f"{label}.kind is unknown: {query.kind}")
                continue
            if query.kind in {"rewrite", "agentic"}:
                if query.max_tokens is None or query.max_tokens <= 0:
                    errors.append(f"{label}.max_tokens must be positive")
                if query.temperature is None or not 0 <= query.temperature <= 2:
                    errors.append(f"{label}.temperature must be between 0 and 2")
                prompts = (
                    (query.prompt, query.answer_prompt, query.compose_prompt)
                    if query.kind == "agentic"
                    else (query.prompt,)
                )
                for prompt in prompts:
                    if prompt is None or not prompt.is_file():
                        errors.append(f"prompt file is missing: {prompt}")

        for pipeline in self.pipelines.values():
            label = f"pipelines.{pipeline.name}"
            if pipeline.kind not in PIPELINE_OUTPUTS:
                errors.append(f"{label}.kind is unknown: {pipeline.kind}")
            if pipeline.query not in self.queries:
                errors.append(f"unknown query: {pipeline.query}")

        for generator in self.generators.values():
            label = f"generators.{generator.name}"
            if not generator.prompt.is_file():
                errors.append(f"prompt file is missing: {generator.prompt}")
            if not 0 <= generator.temperature <= 2:
                errors.append(f"{label}.temperature must be between 0 and 2")
            if generator.max_tokens <= 0 or not 1 <= generator.context_top_k <= 10:
                errors.append(
                    f"{label}: max_tokens must be positive and context_top_k must be 1..10"
                )

        retrieval_references: list[str] = []
        for job in self.generation_jobs.values():
            label = f"generation.{job.name}"
            if not isinstance(job.evaluate, bool):
                errors.append(f"{label}.evaluate must be a boolean")
            if job.task not in {"b", "c"}:
                errors.append(f"{label}.task must be 'b' or 'c'")
            if (job.task == "b") != (job.contexts == "reference"):
                errors.append(
                    f"{label}: Task B uses 'reference'; "
                    "Task C uses a retrieval output"
                )
            if job.generator not in self.generators:
                errors.append(f"unknown generator: {job.generator}")
            if job.task == "c":
                retrieval_references.append(job.contexts)

        for schedule in self.schedules.values():
            retrieval_references.extend(schedule.task_a)
            for job_name in schedule.generation:
                if job_name not in self.generation_jobs:
                    errors.append(f"unknown generation job: {job_name}")

        for reference in retrieval_references:
            pipeline_name, separator, output = reference.partition(".")
            if not separator or "." in output:
                errors.append(
                    f"retrieval output must be '<pipeline>.<output>': {reference!r}"
                )
                continue
            pipeline = self.pipelines.get(pipeline_name)
            if pipeline is None:
                errors.append(f"unknown pipeline: {pipeline_name}")
            elif output not in PIPELINE_OUTPUTS.get(pipeline.kind, ()):
                errors.append(
                    f"pipeline {pipeline.name!r} ({pipeline.kind}) has no output {output!r}"
                )

        errors = list(dict.fromkeys(errors))
        if errors:
            raise ValueError("invalid experiment config:\n- " + "\n- ".join(errors))

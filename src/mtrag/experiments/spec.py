from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping, TypeVar


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


T = TypeVar("T")


def _record(
    defaults: T,
    section: Mapping[str, Any],
    label: str,
    *,
    project_root: Path = Path(),
    paths: Iterable[str] = (),
    urls: Iterable[str] = (),
    positive: Iterable[str] = (),
    required_strings: Iterable[str] = (),
) -> T:
    """Load a flat TOML table without repeating ``section.get`` for every field."""
    path_names, url_names = set(paths), set(urls)
    required_names = set(required_strings)
    values: dict[str, Any] = {}
    for field in fields(defaults):
        name, default = field.name, getattr(defaults, field.name)
        value = section.get(name, default)
        if name in required_names:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label}.{name} must be a non-empty string")
            values[name] = value.strip()
        elif name in path_names:
            values[name] = _path(str(value), project_root)
        else:
            values[name] = type(default)(value)
            if name in url_names:
                values[name] = values[name].rstrip("/")

    invalid = [name for name in positive if values[name] <= 0]
    if invalid:
        raise ValueError(f"{label} values must be positive: {', '.join(invalid)}")
    return replace(defaults, **values)


def _named(
    document: Mapping[str, Any],
    section_name: str,
    parser: Callable[[str, Mapping[str, Any]], T],
) -> tuple[T, ...]:
    section = _section(document, section_name)
    result = []
    for name in section:
        if _SAFE_NAME.fullmatch(name) is None:
            raise ValueError(f"{section_name}.{name} is not a safe identifier")
        result.append(parser(name, _section(section, name)))
    return tuple(result)


def _lookup(items: Iterable[T], name: str, label: str) -> T:
    for item in items:
        if getattr(item, "name") == name:
            return item
    raise ValueError(f"unknown {label}: {name}")


_SAFE_NAME = re.compile(r"[a-z][a-z0-9_]*")
PIPELINE_OUTPUTS = {
    "bge": frozenset(("dense", "sparse", "rrf", "rrf_reranked")),
    "elser": frozenset(("base", "reranked")),
}


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
    prompt: Path | None
    answer_prompt: Path | None
    compose_prompt: Path | None
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

        config = cls(
            path=config_path,
            project_root=project_root,
            run=_record(
                RunConfig(),
                _section(document, "run"),
                "run",
                project_root=project_root,
                paths=("output_root", "benchmark_root"),
                positive=("cpu_slots",),
            ),
            services=_record(
                ServiceConfig(),
                _section(document, "services"),
                "services",
                urls=("elasticsearch_url", "ollama_url"),
            ),
            models=_record(
                ModelConfig(),
                _section(document, "models"),
                "models",
                project_root=project_root,
                paths=("bge_path", "reranker_path"),
                positive=(
                    "bge_batch_size",
                    "bge_max_length",
                    "reranker_batch_size",
                    "reranker_max_length",
                    "ollama_num_ctx",
                    "ollama_timeout",
                ),
            ),
            queries=_named(
                document,
                "queries",
                lambda name, table: _query_config(name, table, project_root),
            ),
            pipelines=_named(
                document,
                "pipelines",
                _pipeline_config,
            ),
            generators=_named(
                document,
                "generators",
                lambda name, table: _generator_config(name, table, project_root),
            ),
            generation_jobs=_named(
                document,
                "generation",
                _generation_job_config,
            ),
            schedules=_named(
                document,
                "schedules",
                _schedule_config,
            ),
            retrieval=_record(
                RetrievalConfig(),
                _section(document, "retrieval"),
                "retrieval",
                positive=(
                    "dense_top_k",
                    "dense_candidate_multiplier",
                    "sparse_top_k",
                    "elser_top_k",
                    "rrf_top_k",
                    "rrf_rank_constant",
                    "prediction_top_k",
                    "request_batch_size",
                ),
                required_strings=("bge_index_revision", "elser_index_revision"),
            ),
            reranking=_record(
                RerankingConfig(),
                _section(document, "reranking"),
                "reranking",
                positive=("input_top_k", "output_top_k", "task_batch_size"),
            ),
            evaluation=_record(
                EvaluationConfig(),
                _section(document, "evaluation"),
                "evaluation",
                positive=("bertscore_batch_size",),
            ),
            thermal=_record(
                ThermalConfig(),
                _section(document, "thermal"),
                "thermal",
            ),
        )
        config.validate()
        return config

    @property
    def default_run_dir(self) -> Path:
        return self.run.output_root / self.run.name

    def query(self, name: str) -> QueryConfig:
        return _lookup(self.queries, name, "query")

    def pipeline(self, name: str) -> PipelineConfig:
        return _lookup(self.pipelines, name, "pipeline")

    def generator(self, name: str) -> GeneratorConfig:
        return _lookup(self.generators, name, "generator")

    def generation_job(self, name: str) -> GenerationJobConfig:
        return _lookup(self.generation_jobs, name, "generation job")

    def schedule(self, name: str) -> ScheduleConfig:
        return _lookup(self.schedules, name, "schedule")

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
        if self.retrieval.prediction_top_k > 10:
            raise ValueError("retrieval.prediction_top_k cannot exceed the official limit 10")
        if self.retrieval.dense_rescore_oversample < 1.0:
            raise ValueError("dense_rescore_oversample must be at least 1.0")
        if self.evaluation.bertscore_model != "microsoft/deberta-xlarge-mnli":
            raise ValueError(
                "the official IBM evaluator requires "
                "evaluation.bertscore_model='microsoft/deberta-xlarge-mnli'"
            )

        for pipeline in self.pipelines:
            self.query(pipeline.query)
        for job in self.generation_jobs:
            self.generator(job.generator)
            if job.task == "c":
                self.resolve_retrieval_output(job.contexts)

        for schedule in self.schedules:
            for reference in schedule.task_a:
                self.resolve_retrieval_output(reference)
            for job_name in schedule.generation:
                self.generation_job(job_name)

        if self.reranking.input_top_k > self.retrieval.rrf_top_k:
            raise ValueError("reranking.input_top_k cannot exceed retrieval.rrf_top_k")
        if self.reranking.output_top_k > self.reranking.input_top_k:
            raise ValueError("reranking.output_top_k cannot exceed reranking.input_top_k")


def _query_config(
    name: str,
    section: Mapping[str, Any],
    project_root: Path,
) -> QueryConfig:
    label = f"queries.{name}"
    kind = _required_string(section, "kind", label)
    if kind not in {
        "last_turn",
        "last_turn_all",
        "gold",
        "rewrite",
        "agentic",
    }:
        raise ValueError(f"{label}.kind is unknown: {kind}")
    if kind not in {"rewrite", "agentic"}:
        return QueryConfig(name, kind, None, None, None, None, None)

    prompt = _prompt(section, label, project_root)
    answer_prompt = (
        _prompt(section, label, project_root, field="answer_prompt")
        if kind == "agentic"
        else None
    )
    compose_prompt = (
        _prompt(section, label, project_root, field="compose_prompt")
        if kind == "agentic"
        else None
    )
    temperature = _temperature(section.get("temperature", 0.0), label)
    max_tokens = int(section.get("max_tokens", 128))
    if max_tokens <= 0:
        raise ValueError(f"{label}.max_tokens must be positive")
    return QueryConfig(
        name=name,
        kind=kind,
        prompt=prompt,
        answer_prompt=answer_prompt,
        compose_prompt=compose_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _pipeline_config(name: str, section: Mapping[str, Any]) -> PipelineConfig:
    label = f"pipelines.{name}"
    kind = _required_string(section, "kind", label)
    if kind not in PIPELINE_OUTPUTS:
        raise ValueError(f"{label}.kind is unknown: {kind}")
    return PipelineConfig(
        name=name,
        kind=kind,
        query=_required_string(section, "query", label),
    )


def _generator_config(
    name: str,
    section: Mapping[str, Any],
    project_root: Path,
) -> GeneratorConfig:
    label = f"generators.{name}"
    max_tokens = int(section.get("max_tokens", 512))
    context_top_k = int(section.get("context_top_k", 5))
    if max_tokens <= 0 or not 1 <= context_top_k <= 10:
        raise ValueError(
            f"{label}: max_tokens must be positive and context_top_k must be 1..10"
        )
    return GeneratorConfig(
        name=name,
        prompt=_prompt(section, label, project_root),
        temperature=_temperature(section.get("temperature", 0.0), label),
        max_tokens=max_tokens,
        context_top_k=context_top_k,
    )


def _generation_job_config(
    name: str,
    section: Mapping[str, Any],
) -> GenerationJobConfig:
    label = f"generation.{name}"
    evaluate = section.get("evaluate", True)
    if not isinstance(evaluate, bool):
        raise ValueError(f"{label}.evaluate must be a boolean")
    task = _required_string(section, "task", label)
    contexts = _required_string(section, "contexts", label)
    if task not in {"b", "c"}:
        raise ValueError(f"{label}.task must be 'b' or 'c'")
    if (task == "b") != (contexts == "reference"):
        raise ValueError(
            f"{label}: Task B uses 'reference'; Task C uses a retrieval output"
        )
    return GenerationJobConfig(
        name,
        task,
        _required_string(section, "generator", label),
        contexts,
        evaluate,
    )


def _schedule_config(name: str, section: Mapping[str, Any]) -> ScheduleConfig:
    label = f"schedules.{name}"
    return ScheduleConfig(
        name,
        _string_tuple(section.get("task_a"), f"{label}.task_a"),
        _string_tuple(section.get("generation"), f"{label}.generation"),
    )


def _prompt(
    section: Mapping[str, Any],
    label: str,
    project_root: Path,
    *,
    field: str = "prompt",
) -> Path:
    path = _path(_required_string(section, field, label), project_root)
    if not path.is_file():
        raise ValueError(f"prompt file is missing: {path}")
    return path


def _temperature(value: Any, label: str) -> float:
    temperature = float(value)
    if not 0.0 <= temperature <= 2.0:
        raise ValueError(f"{label}.temperature must be between 0 and 2")
    return temperature

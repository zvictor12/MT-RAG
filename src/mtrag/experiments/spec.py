from __future__ import annotations

import os
import re
import tomllib
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
class RewriteVariantConfig:
    name: str
    temperature: float


@dataclass(frozen=True, slots=True)
class RewritingConfig:
    max_tokens: int
    variants: tuple[RewriteVariantConfig, ...]

    def variant(self, name: str) -> RewriteVariantConfig:
        for variant in self.variants:
            if variant.name == name:
                return variant
        raise ValueError(f"unknown rewrite variant: {name}")


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
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
    minimum_ndcg5_gain: float
    minimum_improvement_probability: float
    bootstrap_samples: int
    bootstrap_seed: int


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    context_top_k: int
    max_tokens: int
    temperature: float
    run_algorithmic_metrics: bool
    bertscore_model: str
    bertscore_batch_size: int


@dataclass(frozen=True, slots=True)
class ThermalConfig:
    gpu_pause: float
    gpu_resume: float
    gpu_abort: float
    cpu_pause: float
    cpu_resume: float
    cpu_abort: float
    poll_interval: float
    resume_hold: float


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    path: Path
    project_root: Path
    run: RunConfig
    services: ServiceConfig
    models: ModelConfig
    rewriting: RewritingConfig
    retrieval: RetrievalConfig
    reranking: RerankingConfig
    generation: GenerationConfig
    thermal: ThermalConfig

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        config_path = Path(path).expanduser().resolve()
        document = tomllib.loads(config_path.read_text(encoding="utf-8"))
        project_root = config_path.parent.parent

        run = _section(document, "run")
        services = _section(document, "services")
        models = _section(document, "models")
        rewriting = _section(document, "rewriting")
        rewrite_variants = _section(rewriting, "variants")
        if not rewrite_variants:
            rewrite_variants = {
                "qwen_t0": {"temperature": 0.0},
                "qwen_t02": {"temperature": 0.2},
            }
        retrieval = _section(document, "retrieval")
        reranking = _section(document, "reranking")
        generation = _section(document, "generation")
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
            rewriting=RewritingConfig(
                max_tokens=int(rewriting.get("max_tokens", 128)),
                variants=tuple(
                    RewriteVariantConfig(
                        name=str(name),
                        temperature=float(
                            _section(rewrite_variants, str(name)).get(
                                "temperature",
                                0.0,
                            )
                        ),
                    )
                    for name in rewrite_variants
                ),
            ),
            retrieval=RetrievalConfig(
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
                minimum_ndcg5_gain=float(
                    reranking.get("minimum_ndcg5_gain", 0.01)
                ),
                minimum_improvement_probability=float(
                    reranking.get("minimum_improvement_probability", 0.95)
                ),
                bootstrap_samples=int(reranking.get("bootstrap_samples", 5000)),
                bootstrap_seed=int(reranking.get("bootstrap_seed", 42)),
            ),
            generation=GenerationConfig(
                context_top_k=int(generation.get("context_top_k", 5)),
                max_tokens=int(generation.get("max_tokens", 512)),
                temperature=float(generation.get("temperature", 0.0)),
                run_algorithmic_metrics=bool(
                    generation.get("run_algorithmic_metrics", False)
                ),
                bertscore_model=str(
                    generation.get(
                        "bertscore_model",
                        "microsoft/deberta-xlarge-mnli",
                    )
                ),
                bertscore_batch_size=int(generation.get("bertscore_batch_size", 2)),
            ),
            thermal=ThermalConfig(
                gpu_pause=float(thermal.get("gpu_pause", 80.0)),
                gpu_resume=float(thermal.get("gpu_resume", 72.0)),
                gpu_abort=float(thermal.get("gpu_abort", 86.0)),
                cpu_pause=float(thermal.get("cpu_pause", 90.0)),
                cpu_resume=float(thermal.get("cpu_resume", 80.0)),
                cpu_abort=float(thermal.get("cpu_abort", 97.0)),
                poll_interval=float(thermal.get("poll_interval", 5.0)),
                resume_hold=float(thermal.get("resume_hold", 30.0)),
            ),
        )
        config.validate()
        return config

    @property
    def default_run_dir(self) -> Path:
        return self.run.output_root / self.run.name

    def validate(self) -> None:
        positive = {
            "run.cpu_slots": self.run.cpu_slots,
            "models.bge_batch_size": self.models.bge_batch_size,
            "models.bge_max_length": self.models.bge_max_length,
            "models.reranker_batch_size": self.models.reranker_batch_size,
            "models.reranker_max_length": self.models.reranker_max_length,
            "models.ollama_num_ctx": self.models.ollama_num_ctx,
            "models.ollama_timeout": self.models.ollama_timeout,
            "rewriting.max_tokens": self.rewriting.max_tokens,
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
            "reranking.bootstrap_samples": self.reranking.bootstrap_samples,
            "generation.context_top_k": self.generation.context_top_k,
            "generation.max_tokens": self.generation.max_tokens,
            "generation.bertscore_batch_size": self.generation.bertscore_batch_size,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"configuration values must be positive: {', '.join(invalid)}")
        if self.retrieval.prediction_top_k > 10:
            raise ValueError("retrieval.prediction_top_k cannot exceed the official limit 10")
        if self.retrieval.dense_rescore_oversample < 1.0:
            raise ValueError("dense_rescore_oversample must be at least 1.0")
        if not 0.0 <= self.generation.temperature <= 2.0:
            raise ValueError("generation.temperature must be between 0 and 2")
        rewrite_names = [variant.name for variant in self.rewriting.variants]
        if len(rewrite_names) != len(set(rewrite_names)):
            raise ValueError("rewrite variant names must be unique")
        invalid_names = [
            name
            for name in rewrite_names
            if re.fullmatch(r"[a-z][a-z0-9_]*", name) is None
        ]
        if invalid_names:
            raise ValueError("rewrite variant names must be safe identifiers")
        missing_variants = {"qwen_t0", "qwen_t02"} - set(rewrite_names)
        if missing_variants:
            raise ValueError(
                "missing required rewrite variants: "
                + ", ".join(sorted(missing_variants))
            )
        invalid_temperatures = [
            variant.name
            for variant in self.rewriting.variants
            if not 0.0 <= variant.temperature <= 2.0
        ]
        if invalid_temperatures:
            raise ValueError(
                "rewrite temperatures must be between 0 and 2: "
                + ", ".join(invalid_temperatures)
            )
        if self.reranking.input_top_k > self.retrieval.rrf_top_k:
            raise ValueError("reranking.input_top_k cannot exceed retrieval.rrf_top_k")
        if self.reranking.output_top_k > self.reranking.input_top_k:
            raise ValueError("reranking.output_top_k cannot exceed reranking.input_top_k")
        if not 0.0 <= self.reranking.minimum_improvement_probability <= 1.0:
            raise ValueError("minimum_improvement_probability must be between 0 and 1")

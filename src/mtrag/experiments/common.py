from __future__ import annotations

from dataclasses import dataclass

from mtrag.experiments.spec import ExperimentConfig
from mtrag.llm import OllamaClient
from mtrag.runtime import ThermalGuard, ThermalThresholds


def thermal_guard(config: ExperimentConfig) -> ThermalGuard:
    return ThermalGuard(
        gpu=ThermalThresholds(
            config.thermal.gpu_pause,
            config.thermal.gpu_resume,
        ),
        cpu=ThermalThresholds(
            config.thermal.cpu_pause,
            config.thermal.cpu_resume,
        ),
        poll_interval=config.thermal.poll_interval,
        resume_hold=config.thermal.resume_hold,
    )


def ollama_client(config: ExperimentConfig) -> OllamaClient:
    model = config.models
    return OllamaClient(
        url=config.services.ollama_url,
        model=model.ollama_model,
        num_ctx=model.ollama_num_ctx,
        seed=model.ollama_seed,
        keep_alive=model.ollama_keep_alive,
        timeout=model.ollama_timeout,
    )


@dataclass(frozen=True, slots=True)
class OllamaModelInfo:
    identity: str
    provenance: dict[str, str | int]


def ollama_model_info(config: ExperimentConfig) -> OllamaModelInfo:
    model = config.models
    identity = (
        f"{model.ollama_model}@{model.ollama_digest}:"
        f"ctx={model.ollama_num_ctx}:seed={model.ollama_seed}"
    )
    provenance = {
        "model": model.ollama_model,
        "model_digest": model.ollama_digest,
        "num_ctx": model.ollama_num_ctx,
        "seed": model.ollama_seed,
    }
    return OllamaModelInfo(identity, provenance)


def progress(label: str, completed: int, total: int) -> None:
    print(f"{label}: {completed}/{total}", flush=True)

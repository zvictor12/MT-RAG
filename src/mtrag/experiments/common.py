from __future__ import annotations

from dataclasses import replace

from mtrag.config import settings
from mtrag.experiments.spec import ExperimentConfig
from mtrag.llm import OllamaClient
from mtrag.runtime import ThermalGuard, ThermalThresholds


def thermal_guard(config: ExperimentConfig) -> ThermalGuard:
    return ThermalGuard(
        gpu=ThermalThresholds(
            config.thermal.gpu_pause,
            config.thermal.gpu_resume,
            config.thermal.gpu_abort,
        ),
        cpu=ThermalThresholds(
            config.thermal.cpu_pause,
            config.thermal.cpu_resume,
            config.thermal.cpu_abort,
        ),
        poll_interval=config.thermal.poll_interval,
        resume_hold=config.thermal.resume_hold,
    )


def ollama_client(config: ExperimentConfig) -> OllamaClient:
    client_settings = replace(
        settings,
        ollama_url=config.services.ollama_url,
        ollama_model=config.models.ollama_model,
        ollama_num_ctx=config.models.ollama_num_ctx,
        ollama_keep_alive=config.models.ollama_keep_alive,
        ollama_seed=config.models.ollama_seed,
        ollama_timeout=config.models.ollama_timeout,
    )
    return OllamaClient(client_settings)


def chunks[T](values: list[T], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def progress(label: str, completed: int, total: int) -> None:
    print(f"{label}: {completed}/{total}", flush=True)

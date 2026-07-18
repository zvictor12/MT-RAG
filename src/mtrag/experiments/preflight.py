from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import requests

from mtrag.data.benchmark import DOMAINS
from mtrag.experiments.artifacts import RunArtifacts
from mtrag.experiments.common import ollama_client
from mtrag.experiments.spec import ExperimentConfig


_BGE_FILES = (
    "config.json",
    "pytorch_model.bin",
    "colbert_linear.pt",
    "sparse_linear.pt",
    "tokenizer.json",
)
_RERANKER_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
)


def _require_files(label: str, directory: Path, names: tuple[str, ...]) -> None:
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        raise RuntimeError(
            f"{label} model is incomplete in {directory}: {', '.join(missing)}"
        )


def _get_json(url: str, path: str, *, timeout: int = 15) -> Any:
    response = requests.get(f"{url}{path}", timeout=timeout)
    response.raise_for_status()
    return response.json()


class StageLike(Protocol):
    kind: str
    params: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PreflightRequirements:
    generation_tasks: bool
    bge_model: bool
    reranker: bool
    ollama: bool
    cuda: bool
    bge_modes: frozenset[str]
    elser: bool


def requirements_for(
    stages: Iterable[StageLike] | None,
) -> PreflightRequirements:
    if stages is None:
        kinds = {
            "encode",
            "rerank",
            "rewrite",
            "generate",
            "evaluate_generation_batch",
        }
        methods = {"dense", "sparse", "elser"}
        generation_tasks = True
    else:
        planned = tuple(stages)
        kinds = {stage.kind for stage in planned}
        methods = {
            str(stage.params.get("method"))
            for stage in planned
            if stage.kind == "retrieve"
        }
        generation_tasks = bool(planned)
    return PreflightRequirements(
        generation_tasks=generation_tasks,
        bge_model="encode" in kinds,
        reranker="rerank" in kinds,
        ollama=bool(kinds & {"rewrite", "generate"}),
        cuda=bool(
            kinds & {"encode", "rerank", "evaluate_generation_batch"}
        ),
        bge_modes=frozenset(methods & {"dense", "sparse"}),
        elser="elser" in methods,
    )


def _bge_indices(modes: Iterable[str]) -> set[str]:
    return {
        f"mtrag-{domain}-bge-m3-{mode}"
        for domain in DOMAINS
        for mode in modes
    }


def _validate_bge_mapping(index: str, mapping: Mapping[str, Any]) -> None:
    embedding = mapping[index]["mappings"]["properties"]["embedding"]
    if index.endswith("-dense"):
        expected = {
            "type": "dense_vector",
            "dims": 1024,
            "similarity": "dot_product",
            "index_options.type": "int8_hnsw",
        }
    else:
        expected = {"type": "sparse_vector"}
    actual = dict(embedding)
    actual["index_options.type"] = embedding.get("index_options", {}).get("type")
    mismatched = {
        key: (actual.get(key), value)
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatched:
        raise RuntimeError(f"incompatible mapping for {index}: {mismatched}")


def _check_elasticsearch(
    config: ExperimentConfig,
    *,
    bge_modes: frozenset[str],
    elser: bool,
) -> str:
    url = config.services.elasticsearch_url
    info = _get_json(url, "")
    rows = _get_json(
        url,
        "/_cat/indices/mtrag-*?format=json&h=index,docs.count",
    )
    counts = {row["index"]: int(row["docs.count"]) for row in rows}

    required = _bge_indices(bge_modes)
    if elser:
        required.update(f"mtrag-{domain}-elser" for domain in DOMAINS)
    unavailable = {
        index: counts.get(index)
        for index in sorted(required)
        if counts.get(index, 0) <= 0
    }
    if unavailable:
        raise RuntimeError(f"Elasticsearch indices are not ready: {unavailable}")

    bge_indices = _bge_indices(bge_modes)
    if bge_indices:
        mapping = _get_json(url, "/mtrag-*-bge-m3-*/_mapping")
        for index in sorted(bge_indices):
            _validate_bge_mapping(index, mapping)
    if bge_modes == {"dense", "sparse"}:
        for domain in DOMAINS:
            dense = counts[f"mtrag-{domain}-bge-m3-dense"]
            sparse = counts[f"mtrag-{domain}-bge-m3-sparse"]
            if dense != sparse:
                raise RuntimeError(
                    f"BGE index counts differ for {domain}: "
                    f"dense={dense}, sparse={sparse}"
                )

    return str(info["version"]["number"])


def _check_elser_endpoint(config: ExperimentConfig) -> bool:
    try:
        response = requests.post(
            (
                f"{config.services.elasticsearch_url}/_inference/sparse_embedding/"
                f"{config.services.elser_inference_id}"
            ),
            json={"input": "preflight query"},
            timeout=120,
        )
    except requests.RequestException:
        return False
    return response.ok


def _check_cuda() -> str:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is missing; run `make sync-ml`") from error
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot use CUDA")
    return torch.cuda.get_device_name(0)


def preflight(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
    *,
    stages: Iterable[StageLike] | None = None,
) -> None:
    required = requirements_for(stages)
    artifacts.create_directories()
    generation_tasks = (
        config.run.benchmark_root
        / "mtrag-human"
        / "generation_tasks"
        / "reference.jsonl"
    )
    if required.generation_tasks and not generation_tasks.is_file():
        raise RuntimeError(f"benchmark is missing: {generation_tasks}")

    ready = ["benchmark"]
    if required.bge_model:
        _require_files("BGE-M3", config.models.bge_path, _BGE_FILES)
        ready.append("BGE-M3")
    if required.reranker:
        _require_files("reranker", config.models.reranker_path, _RERANKER_FILES)
        ready.append("reranker")
    if required.cuda:
        ready.append(_check_cuda())
    if required.bge_modes or required.elser:
        version = _check_elasticsearch(
            config,
            bge_modes=required.bge_modes,
            elser=required.elser,
        )
        ready.append(f"Elasticsearch {version}")
        if required.bge_modes:
            ready.append(f"BGE index {config.retrieval.bge_index_revision}")
    if required.elser and not _check_elser_endpoint(config):
        raise RuntimeError(
            f"ELSER inference endpoint {config.services.elser_inference_id!r} "
            "is not ready; unload Ollama and run scripts/setup_elasticsearch.py"
        )
    if required.elser:
        ready.append(f"ELSER index {config.retrieval.elser_index_revision}")

    if required.ollama:
        client = ollama_client(config)
        installed = client.installed_model_digests()
        if config.models.ollama_model not in installed:
            raise RuntimeError(
                f"Ollama model {config.models.ollama_model!r} is not installed"
            )
        actual_digest = installed[config.models.ollama_model]
        if config.models.ollama_digest and actual_digest != config.models.ollama_digest:
            raise RuntimeError(
                f"Ollama digest changed for {config.models.ollama_model}: "
                f"expected {config.models.ollama_digest}, got {actual_digest}"
            )
        ready.append(config.models.ollama_model)

    print(f"ready: {', '.join(ready)}", flush=True)

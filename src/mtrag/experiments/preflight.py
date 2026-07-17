from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

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


def _bge_indices() -> set[str]:
    return {
        f"mtrag-{domain}-bge-m3-{mode}"
        for domain in DOMAINS
        for mode in ("dense", "sparse")
    }


def _validate_bge_mapping(index: str, mapping: Mapping[str, Any]) -> None:
    embedding = mapping[index]["mappings"]["properties"]["embedding"]
    if index.endswith("-dense"):
        expected = {
            "type": "dense_vector",
            "dims": 1024,
            "similarity": "dot_product",
        }
    else:
        expected = {"type": "sparse_vector"}
    mismatched = {
        key: (embedding.get(key), value)
        for key, value in expected.items()
        if embedding.get(key) != value
    }
    if mismatched:
        raise RuntimeError(f"incompatible mapping for {index}: {mismatched}")
    if index.endswith("-dense"):
        index_type = embedding.get("index_options", {}).get("type")
        if index_type != "int8_hnsw":
            raise RuntimeError(
                f"incompatible mapping for {index}: index_options.type={index_type}"
            )


def _check_elasticsearch(config: ExperimentConfig) -> tuple[str, bool]:
    url = config.services.elasticsearch_url
    info = _get_json(url, "")
    rows = _get_json(
        url,
        "/_cat/indices/mtrag-*?format=json&h=index,docs.count",
    )
    counts = {row["index"]: int(row["docs.count"]) for row in rows}

    required = _bge_indices()
    missing = sorted(required - counts.keys())
    empty = sorted(index for index in required if counts.get(index, 0) <= 0)
    if missing or empty:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if empty:
            details.append(f"empty={empty}")
        raise RuntimeError("BGE Elasticsearch indices are not ready: " + "; ".join(details))

    mapping = _get_json(url, "/mtrag-*-bge-m3-*/_mapping")
    for index in sorted(required):
        _validate_bge_mapping(index, mapping)
    for domain in DOMAINS:
        dense = counts[f"mtrag-{domain}-bge-m3-dense"]
        sparse = counts[f"mtrag-{domain}-bge-m3-sparse"]
        if dense != sparse:
            raise RuntimeError(
                f"BGE index counts differ for {domain}: dense={dense}, sparse={sparse}"
            )

    elser_indices = {f"mtrag-{domain}-elser" for domain in DOMAINS}
    elser_ready = elser_indices.issubset(counts)
    return str(info["version"]["number"]), elser_ready


def _check_elser_endpoint(config: ExperimentConfig) -> bool:
    # The restored semantic_text mapping expects the endpoint name from .env.
    # It is intentionally a warning here: the independent BGE branch can run
    # while a still-running Kaggle ELSER snapshot is being downloaded.
    try:
        response = requests.get(
            (
                f"{config.services.elasticsearch_url}/_inference/sparse_embedding/"
                f"{config.services.elser_inference_id}"
            ),
            timeout=10,
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
        raise RuntimeError("PyTorch cannot use CUDA; run `make diagnose`")
    return torch.cuda.get_device_name(0)


def preflight(config: ExperimentConfig, artifacts: RunArtifacts) -> None:
    artifacts.create_directories()
    generation_tasks = (
        config.run.benchmark_root
        / "mtrag-human"
        / "generation_tasks"
        / "reference.jsonl"
    )
    if not generation_tasks.is_file():
        raise RuntimeError(f"benchmark is missing: {generation_tasks}")

    _require_files("BGE-M3", config.models.bge_path, _BGE_FILES)
    _require_files("reranker", config.models.reranker_path, _RERANKER_FILES)
    device = _check_cuda()
    elasticsearch_version, elser_indices_ready = _check_elasticsearch(config)
    elser_endpoint_ready = _check_elser_endpoint(config)

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

    print(
        f"ready: benchmark, Elasticsearch {elasticsearch_version}, "
        f"{config.models.ollama_model}, BGE-M3, reranker, {device}",
        flush=True,
    )
    if not (elser_indices_ready and elser_endpoint_ready):
        print(
            "warning: ELSER is not ready; run the BGE phase now, then extend "
            "the same run with the full phase after `make elser-setup`",
            flush=True,
        )

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import requests

from mtrag.data.benchmark import DOMAINS
from mtrag.experiments.common import ollama_client
from mtrag.experiments.spec import ExperimentConfig


class PreflightError(RuntimeError):
    pass


class StageLike(Protocol):
    kind: str
    params: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PreflightRequirements:
    cuda: bool
    ollama: bool
    bge_modes: frozenset[str]
    elser: bool


def requirements_for(stages: Iterable[StageLike]) -> PreflightRequirements:
    planned = tuple(stages)
    kinds = {stage.kind for stage in planned}
    methods = {
        str(stage.params.get("method"))
        for stage in planned
        if stage.kind == "retrieve"
    }
    return PreflightRequirements(
        cuda=bool(kinds & {"encode", "rerank", "evaluate_generation_batch"}),
        ollama=bool(kinds & {"rewrite", "generate"}),
        bge_modes=frozenset(methods & {"dense", "sparse"}),
        elser="elser" in methods,
    )


def _get_json(url: str, path: str, *, timeout: int = 15) -> Any:
    response = requests.get(f"{url}{path}", timeout=timeout)
    response.raise_for_status()
    return response.json()


def _mapping_issue(index: str, mapping: Mapping[str, Any]) -> str | None:
    embedding = mapping[index]["mappings"]["properties"]["embedding"]
    expected = (
        {
            "type": "dense_vector",
            "dims": 1024,
            "similarity": "dot_product",
            "index_options.type": "int8_hnsw",
        }
        if index.endswith("-dense")
        else {"type": "sparse_vector"}
    )
    actual = dict(embedding)
    actual["index_options.type"] = embedding.get("index_options", {}).get("type")
    mismatched = {
        key: (actual.get(key), value)
        for key, value in expected.items()
        if actual.get(key) != value
    }
    return f"incompatible mapping for {index}: {mismatched}" if mismatched else None


def _check_elasticsearch(
    config: ExperimentConfig,
    required: PreflightRequirements,
) -> tuple[str, list[str]]:
    url = config.services.elasticsearch_url
    info = _get_json(url, "")
    rows = _get_json(url, "/_cat/indices/mtrag-*?format=json&h=index,docs.count")
    counts = {row["index"]: int(row["docs.count"]) for row in rows}
    bge_indices = {
        f"mtrag-{domain}-bge-m3-{mode}"
        for domain in DOMAINS
        for mode in required.bge_modes
    }
    indices = set(bge_indices)
    if required.elser:
        indices.update(f"mtrag-{domain}-elser" for domain in DOMAINS)

    errors = []
    unavailable = {
        index: counts.get(index)
        for index in sorted(indices)
        if counts.get(index, 0) <= 0
    }
    if unavailable:
        errors.append(f"Elasticsearch indices are not ready: {unavailable}")

    if bge_indices:
        mapping = _get_json(url, "/mtrag-*-bge-m3-*/_mapping")
        errors.extend(
            issue
            for index in sorted(bge_indices & mapping.keys())
            if (issue := _mapping_issue(index, mapping)) is not None
        )
    if required.bge_modes == {"dense", "sparse"}:
        for domain in DOMAINS:
            dense = counts.get(f"mtrag-{domain}-bge-m3-dense")
            sparse = counts.get(f"mtrag-{domain}-bge-m3-sparse")
            if dense and sparse and dense != sparse:
                errors.append(
                    f"BGE index counts differ for {domain}: "
                    f"dense={dense}, sparse={sparse}"
                )
    return str(info["version"]["number"]), errors


def _elser_ready(config: ExperimentConfig) -> bool:
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


def _cuda_status() -> tuple[str | None, str | None]:
    try:
        import torch
    except ImportError:
        return None, "PyTorch is missing; run `make sync-ml`"
    if not torch.cuda.is_available():
        return None, "PyTorch cannot use CUDA"
    return torch.cuda.get_device_name(0), None


def preflight(config: ExperimentConfig, *, stages: Iterable[StageLike]) -> None:
    required = requirements_for(stages)
    ready: list[str] = []
    errors: list[str] = []

    benchmark = config.run.benchmark_root / "mtrag-human/generation_tasks/reference.jsonl"
    if benchmark.is_file():
        ready.append("benchmark")
    else:
        errors.append(f"benchmark is missing: {benchmark}")
    if required.cuda:
        device, error = _cuda_status()
        if error:
            errors.append(error)
        else:
            ready.append(device or "CUDA")
    if required.bge_modes or required.elser:
        try:
            version, es_errors = _check_elasticsearch(config, required)
        except (requests.RequestException, KeyError, TypeError, ValueError) as error:
            errors.append(f"Elasticsearch preflight failed: {error}")
        else:
            errors.extend(es_errors)
            ready.append(f"Elasticsearch {version}")
            if required.bge_modes and not es_errors:
                ready.append(f"BGE index {config.retrieval.bge_index_revision}")
            if required.elser and not es_errors:
                ready.append(f"ELSER index {config.retrieval.elser_index_revision}")
    if required.elser and not _elser_ready(config):
        errors.append(
            f"ELSER inference endpoint {config.services.elser_inference_id!r} "
            "is not ready; unload Ollama and run scripts/setup_elasticsearch.py"
        )

    if required.ollama:
        try:
            installed = ollama_client(config).installed_model_digests()
            actual = installed.get(config.models.ollama_model)
        except (requests.RequestException, KeyError, TypeError, ValueError) as error:
            errors.append(f"Ollama preflight failed: {error}")
        else:
            if actual is None:
                errors.append(
                    f"Ollama model {config.models.ollama_model!r} is not installed"
                )
            elif config.models.ollama_digest and actual != config.models.ollama_digest:
                errors.append(
                    f"Ollama digest changed for {config.models.ollama_model}: "
                    f"expected {config.models.ollama_digest}, got {actual}"
                )
            else:
                ready.append(config.models.ollama_model)

    if errors:
        raise PreflightError("preflight failed:\n- " + "\n- ".join(errors))
    print(f"ready: {', '.join(ready)}", flush=True)

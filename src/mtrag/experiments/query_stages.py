from __future__ import annotations

from pathlib import Path

from mtrag.data import BenchmarkRepository
from mtrag.encoding import BGEM3QueryEncoder, BgeFeatureStore
from mtrag.experiments.artifacts import (
    JsonlCheckpoint,
    RunArtifacts,
    read_jsonl,
    write_jsonl_atomic,
)
from mtrag.experiments.common import ollama_client, progress, thermal_guard
from mtrag.experiments.spec import ExperimentConfig
from mtrag.llm import QueryRewriter
from mtrag.llm.prompts import REWRITE_PROMPT_VERSION
from mtrag.runtime import SqliteCache, stable_key
from mtrag.schemas import BenchmarkTask, BgeFeatures, QueryCase, QueryVariant


SINGLE_TURN_REWRITE_VERSION = "single-turn-identity-v1"
QWEN_T0 = "qwen_t0"
QWEN_T02 = "qwen_t02"


def _final_user_question(task: BenchmarkTask) -> str:
    for message in reversed(task.messages):
        if message.speaker == "user":
            return message.text
    raise ValueError(f"task {task.task_id} has no user message")


def _single_turn_record(
    task: BenchmarkTask,
    *,
    variant: str = QWEN_T0,
    temperature: float = 0.0,
) -> dict:
    return {
        "task_id": task.task_id,
        "query": _final_user_question(task),
        "rewrite_method": "identity",
        "rewrite_version": SINGLE_TURN_REWRITE_VERSION,
        "rewrite_variant": variant,
        "temperature": temperature,
    }


def _prepare_rewrite_checkpoint(
    path: Path,
    tasks: tuple[BenchmarkTask, ...],
    *,
    variant: str,
    temperature: float,
) -> JsonlCheckpoint:
    checkpoint = JsonlCheckpoint(path)
    tasks_by_id = {task.task_id: task for task in tasks}
    records = []
    changed = False

    for task_id, record in checkpoint.records.items():
        task = tasks_by_id.get(task_id)
        if task is not None and task.turn == 1:
            identity = _single_turn_record(
                task,
                variant=variant,
                temperature=temperature,
            )
            records.append(identity)
            changed = changed or record != identity
        else:
            recorded_variant = record.get("rewrite_variant")
            recorded_temperature = record.get("temperature")
            if recorded_variant not in {None, variant}:
                raise ValueError(
                    f"rewrite variant mismatch in {path}: "
                    f"{recorded_variant!r} != {variant!r}"
                )
            if (
                recorded_temperature is not None
                and float(recorded_temperature) != temperature
            ):
                raise ValueError(
                    f"rewrite temperature mismatch in {path}: "
                    f"{recorded_temperature!r} != {temperature!r}"
                )
            normalized = dict(record)
            normalized["rewrite_method"] = "qwen"
            normalized["rewrite_version"] = normalized.pop(
                "prompt_version",
                REWRITE_PROMPT_VERSION,
            )
            normalized["rewrite_variant"] = variant
            normalized["temperature"] = temperature
            records.append(normalized)
            changed = changed or record != normalized

    if not changed:
        return checkpoint

    write_jsonl_atomic(path, records)
    print("normalized rewrites and restored original turn-1 text", flush=True)
    return JsonlCheckpoint(path)


def _temperature(config: ExperimentConfig, name: str, default: float) -> float:
    variants = getattr(config.rewriting, "variants", ())
    for variant in variants:
        if variant.name == name:
            return float(variant.temperature)
    return default


def _rewrite_variant(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
    *,
    variant: str,
    temperature: float,
    output: Path,
) -> None:
    repository = BenchmarkRepository(config.run.benchmark_root)
    tasks = repository.load_tasks()
    checkpoint = _prepare_rewrite_checkpoint(
        output,
        tasks,
        variant=variant,
        temperature=temperature,
    )
    pending = [task for task in tasks if task.task_id not in checkpoint.completed]
    client = ollama_client(config)

    with SqliteCache(artifacts.cache) as cache:
        rewriter = QueryRewriter(
            client,
            model_name=(
                f"{config.models.ollama_model}@{config.models.ollama_digest}"
            ),
            cache=cache,
            guard=thermal_guard(config),
            max_tokens=config.rewriting.max_tokens,
            temperature=temperature,
        )
        try:
            for index, task in enumerate(pending, start=1):
                if task.turn == 1:
                    record = _single_turn_record(
                        task,
                        variant=variant,
                        temperature=temperature,
                    )
                else:
                    record = {
                        "task_id": task.task_id,
                        "query": rewriter.rewrite(task),
                        "rewrite_method": "qwen",
                        "rewrite_variant": variant,
                        "temperature": temperature,
                        "model": config.models.ollama_model,
                        "model_digest": config.models.ollama_digest,
                        "max_tokens": config.rewriting.max_tokens,
                        "rewrite_version": REWRITE_PROMPT_VERSION,
                    }
                checkpoint.append(record)
                if index % 10 == 0 or index == len(pending):
                    progress(
                        f"rewrite {variant}",
                        len(tasks) - len(pending) + index,
                        len(tasks),
                    )
        finally:
            client.unload()


def rewrite_qwen(config: ExperimentConfig, artifacts: RunArtifacts) -> None:
    _rewrite_variant(
        config,
        artifacts,
        variant=QWEN_T0,
        temperature=_temperature(config, QWEN_T0, 0.0),
        output=artifacts.qwen_queries,
    )


def _migrate_qwen_t0(config: ExperimentConfig, artifacts: RunArtifacts) -> None:
    destination = artifacts.rewrite_queries(QWEN_T0)
    if destination.exists():
        return
    records = read_jsonl(artifacts.qwen_queries)
    tasks = BenchmarkRepository(config.run.benchmark_root).load_tasks()
    tasks_by_id = {task.task_id: task for task in tasks}
    temperature = _temperature(config, QWEN_T0, 0.0)
    normalized = []
    for record in records:
        task = tasks_by_id[record["task_id"]]
        if task.turn == 1:
            normalized.append(
                _single_turn_record(
                    task,
                    variant=QWEN_T0,
                    temperature=temperature,
                )
            )
        else:
            item = dict(record)
            item["rewrite_method"] = "qwen"
            item["rewrite_variant"] = QWEN_T0
            item["temperature"] = temperature
            item["max_tokens"] = config.rewriting.max_tokens
            item["rewrite_version"] = item.pop(
                "prompt_version",
                item.get("rewrite_version", REWRITE_PROMPT_VERSION),
            )
            normalized.append(item)
    write_jsonl_atomic(destination, normalized)
    print(f"migrated legacy rewrites to {destination}", flush=True)


def rewrite_qwen_t02(config: ExperimentConfig, artifacts: RunArtifacts) -> None:
    _migrate_qwen_t0(config, artifacts)
    temperature = _temperature(config, QWEN_T02, 0.2)
    _rewrite_variant(
        config,
        artifacts,
        variant=QWEN_T02,
        temperature=temperature,
        output=artifacts.rewrite_queries(QWEN_T02),
    )


def feature_key(case: QueryCase) -> str:
    return f"{case.variant.value}:{case.task_id}"


def load_query_cases(
    repository: BenchmarkRepository,
    qwen_path: Path,
) -> list[QueryCase]:
    cases: list[QueryCase] = []
    cases.extend(repository.query_cases(QueryVariant.LAST))
    cases.extend(
        repository.query_cases(QueryVariant.QWEN, qwen_queries=qwen_path)
    )
    cases.extend(repository.query_cases(QueryVariant.GOLD))
    return cases


def encode_bge(config: ExperimentConfig, artifacts: RunArtifacts) -> None:
    repository = BenchmarkRepository(config.run.benchmark_root)
    cases = load_query_cases(repository, artifacts.qwen_queries)
    _encode_cases(config, artifacts, cases, merge=False)


def encode_bge_variants(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
) -> None:
    repository = BenchmarkRepository(config.run.benchmark_root)
    cases = repository.query_cases(
        QueryVariant.QWEN_T02,
        qwen_queries=artifacts.rewrite_queries(QWEN_T02),
    )
    _encode_cases(config, artifacts, cases, merge=True)


def _encode_cases(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
    cases: list[QueryCase] | tuple[QueryCase, ...],
    *,
    merge: bool,
) -> None:
    namespace = "bge_query"

    by_text: dict[str, list[QueryCase]] = {}
    for case in cases:
        by_text.setdefault(case.text, []).append(case)

    with SqliteCache(artifacts.cache) as cache:
        missing: list[tuple[str, str]] = []
        for text in by_text:
            key = stable_key(
                config.models.bge_revision,
                config.models.bge_max_length,
                text,
            )
            if cache.get(namespace, key) is None:
                missing.append((key, text))

        if missing:
            encoder = BGEM3QueryEncoder(
                config.models.bge_path,
                batch_size=config.models.bge_batch_size,
                max_length=config.models.bge_max_length,
                guard=thermal_guard(config),
            )
            try:
                checkpoint_size = max(256, config.models.bge_batch_size * 8)
                for start in range(0, len(missing), checkpoint_size):
                    batch = missing[start : start + checkpoint_size]
                    encoded = encoder.encode([text for _key, text in batch])
                    cache.put_many(
                        namespace,
                        {
                            key: {
                                "dense": list(features.dense),
                                "sparse": features.sparse,
                            }
                            for (key, _text), features in zip(
                                batch,
                                encoded,
                                strict=True,
                            )
                        },
                    )
                    progress(
                        "bge encode variants" if merge else "bge encode",
                        min(start + len(batch), len(missing)),
                        len(missing),
                    )
            finally:
                encoder.close()

        features_by_case = (
            BgeFeatureStore(artifacts.bge_features).load() if merge else {}
        )
        for text, text_cases in by_text.items():
            key = stable_key(
                config.models.bge_revision,
                config.models.bge_max_length,
                text,
            )
            value = cache.get(namespace, key)
            if value is None:
                raise RuntimeError(
                    f"missing cached BGE feature for {text_cases[0].task_id}"
                )
            features = BgeFeatures(
                dense=tuple(float(number) for number in value["dense"]),
                sparse={
                    str(token): float(weight)
                    for token, weight in value["sparse"].items()
                },
            )
            for case in text_cases:
                features_by_case[feature_key(case)] = features

    BgeFeatureStore(artifacts.bge_features).save(features_by_case)
    print(f"saved {len(features_by_case)} query feature sets", flush=True)

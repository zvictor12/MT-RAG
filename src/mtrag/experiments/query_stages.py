from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from mtrag.data import BenchmarkRepository
from mtrag.encoding import BGEM3QueryEncoder, BgeFeatureStore
from mtrag.experiments.artifacts import JsonlCheckpoint, RunArtifacts
from mtrag.experiments.common import (
    ollama_client,
    ollama_model_info,
    progress,
    thermal_guard,
)
from mtrag.experiments.spec import ExperimentConfig
from mtrag.llm import HistoryQueryAgent, QueryRewriter
from mtrag.llm.prompts import PromptTemplate
from mtrag.runtime import SqliteCache, stable_key
from mtrag.schemas import BenchmarkTask, BgeFeatures, QueryCase, QueryVariant


SINGLE_TURN_REWRITE_VERSION = "single-turn-identity-v1"
BGE_QUERY_CACHE_VERSION = "flagembedding-bge-m3-v1"


def query_cases(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
    query_name: str,
    query_revision: str,
) -> tuple[QueryCase, ...]:
    query = config.query(query_name)
    repository = BenchmarkRepository(config.run.benchmark_root)
    if query.kind == "last_turn_all":
        return repository.all_task_last_query_cases()
    variant = {
        "last_turn": QueryVariant.LAST,
        "gold": QueryVariant.GOLD,
    }.get(query.kind)
    if variant is not None:
        return repository.query_cases(variant)
    return repository.query_cases(
        QueryVariant.QWEN,
        qwen_queries=artifacts.rewrite(query_name, query_revision),
    )


def rewrite_query(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
    *,
    query_name: str,
    query_revision: str,
) -> None:
    query = config.query(query_name)
    tasks = BenchmarkRepository(config.run.benchmark_root).load_tasks()
    checkpoint = JsonlCheckpoint(artifacts.rewrite(query_name, query_revision))
    pending = [task for task in tasks if task.task_id not in checkpoint.completed]
    if not pending:
        if query.kind == "agentic":
            _validate_agentic_rewrites(tasks, checkpoint.records)
        return

    client = ollama_client(config)
    prompt = PromptTemplate.from_file(query.prompt)
    answer_prompt = (
        PromptTemplate.from_file(query.answer_prompt)
        if getattr(query, "answer_prompt", None) is not None
        else None
    )
    compose_prompt = (
        PromptTemplate.from_file(query.compose_prompt)
        if getattr(query, "compose_prompt", None) is not None
        else None
    )
    model_identity, model_provenance = ollama_model_info(config)
    provenance = {
        "rewrite_variant": query_name,
        "temperature": query.temperature,
        **model_provenance,
    }

    with SqliteCache(artifacts.cache) as cache:
        common = {
            "model_name": model_identity,
            "cache": cache,
            "guard": thermal_guard(config),
            "max_tokens": query.max_tokens,
            "temperature": query.temperature,
        }
        if query.kind == "agentic":
            rewriter = HistoryQueryAgent(
                client,
                question_prompt=prompt,
                answer_prompt=answer_prompt,
                composition_prompt=compose_prompt,
                **common,
            )
            prompt_revision = ":".join(
                (prompt.sha256, answer_prompt.sha256, compose_prompt.sha256)
            )
        else:
            rewriter = QueryRewriter(client, prompt=prompt, **common)
            prompt_revision = prompt.sha256
        try:
            for index, task in enumerate(pending, start=1):
                checkpoint.append(
                    _rewrite_record(
                        task,
                        rewriter,
                        prompt_revision,
                        provenance,
                        agentic=query.kind == "agentic",
                    )
                )
                if index % 10 == 0 or index == len(pending):
                    progress(
                        f"rewrite {query_name}",
                        len(tasks) - len(pending) + index,
                        len(tasks),
                    )
        finally:
            client.unload()
    if query.kind == "agentic":
        _validate_agentic_rewrites(tasks, checkpoint.records)


def _rewrite_record(
    task: BenchmarkTask,
    rewriter: QueryRewriter | HistoryQueryAgent,
    prompt_revision: str,
    provenance: dict,
    *,
    agentic: bool,
) -> dict:
    details = {}
    if task.turn == 1:
        query = _final_user_question(task)
        method = "identity"
        revision = SINGLE_TURN_REWRITE_VERSION
    elif agentic:
        outcome = rewriter.rewrite(task)
        query = outcome.query
        method = "history_agent"
        revision = prompt_revision
        details["resolution"] = outcome.resolution
        details["clarification_questions"] = outcome.questions
        details["resolution_status"] = outcome.status
        details["evidence_ids"] = outcome.evidence_ids
        if outcome.composition is not None:
            details["composition"] = outcome.composition
    else:
        query = rewriter.rewrite(task)
        method = "qwen"
        revision = prompt_revision
    return {
        "task_id": task.task_id,
        "query": query,
        "rewrite_method": method,
        "rewrite_version": revision,
        **details,
        **provenance,
    }


def encode_bge_query(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
    *,
    query_name: str,
    query_revision: str,
    feature_revision: str,
) -> None:
    cases = query_cases(config, artifacts, query_name, query_revision)
    features = _encode_texts(config, cases, artifacts.cache)
    BgeFeatureStore(
        artifacts.bge_features(query_name, feature_revision)
    ).save(features)
    print(f"saved {len(features)} BGE query feature sets", flush=True)


def _encode_texts(
    config: ExperimentConfig,
    cases: Sequence[QueryCase],
    cache_path: Path,
) -> dict[str, BgeFeatures]:
    texts = tuple(dict.fromkeys(case.text for case in cases))
    namespace = "bge_query"
    with SqliteCache(cache_path) as cache:
        keys = {
            text: stable_key(
                BGE_QUERY_CACHE_VERSION,
                config.models.bge_revision,
                config.models.bge_max_length,
                text,
            )
            for text in texts
        }
        features: dict[str, BgeFeatures] = {}
        missing: list[str] = []
        for text, key in keys.items():
            value = cache.get(namespace, key)
            if value is None:
                missing.append(text)
            else:
                features[text] = _cached_features(value)

        if missing:
            with BGEM3QueryEncoder(
                config.models.bge_path,
                batch_size=config.models.bge_batch_size,
                max_length=config.models.bge_max_length,
                guard=thermal_guard(config),
            ) as encoder:
                step = max(256, config.models.bge_batch_size * 8)
                for start in range(0, len(missing), step):
                    batch = missing[start : start + step]
                    encoded = encoder.encode(batch)
                    features.update(zip(batch, encoded, strict=True))
                    cache.put_many(
                        namespace,
                        {
                            keys[text]: {
                                "dense": list(value.dense),
                                "sparse": value.sparse,
                            }
                            for text, value in zip(batch, encoded, strict=True)
                        },
                    )
                    progress(
                        "bge encode",
                        min(start + len(batch), len(missing)),
                        len(missing),
                    )
    return {case.task_id: features[case.text] for case in cases}


def _cached_features(value) -> BgeFeatures:
    return BgeFeatures(
        dense=tuple(float(number) for number in value["dense"]),
        sparse={
            str(token): float(weight)
            for token, weight in value["sparse"].items()
        },
    )


def _final_user_question(task: BenchmarkTask) -> str:
    return next(
        message.text
        for message in reversed(task.messages)
        if message.speaker == "user"
    )


def _validate_agentic_rewrites(
    tasks: Sequence[BenchmarkTask],
    records: dict[str, dict],
) -> None:
    expected = {task.task_id for task in tasks}
    actual = set(records)
    if actual != expected:
        raise RuntimeError(
            f"agentic rewrite is incomplete: {len(actual)}/{len(expected)}"
        )

    statuses = Counter(
        record.get("resolution_status", "identity")
        for record in records.values()
    )
    invalid = set(statuses) - {"identity", "standalone", "resolved"}
    if invalid:
        raise RuntimeError(
            "agentic rewrite contains invalid statuses: "
            + ", ".join(sorted(invalid))
        )

    for task in tasks:
        record = records[task.task_id]
        status = record.get("resolution_status", "identity")
        if task.turn == 1 and status != "identity":
            raise RuntimeError(f"turn 1 was rewritten: {task.task_id}")
        if task.turn > 1 and status == "identity":
            raise RuntimeError(f"multi-turn query bypassed agent: {task.task_id}")

    print(
        "agentic rewrite: "
        + ", ".join(f"{key}={statuses[key]}" for key in sorted(statuses)),
        flush=True,
    )

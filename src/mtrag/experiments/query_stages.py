from __future__ import annotations

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
from mtrag.schemas import ArtifactRef, BgeFeatures, QueryCase, QueryVariant


SINGLE_TURN_REWRITE_VERSION = "single-turn-identity-v1"
BGE_QUERY_CACHE_VERSION = "flagembedding-bge-m3-v1"


def load_query_cases(
    repository: BenchmarkRepository,
    config: ExperimentConfig,
    artifacts: RunArtifacts,
    query: ArtifactRef,
) -> tuple[QueryCase, ...]:
    query_config = config.query(query.name)
    if query_config.kind == "last_turn_all":
        return repository.all_task_last_query_cases()
    variant = {
        "last_turn": QueryVariant.LAST,
        "gold": QueryVariant.GOLD,
    }.get(query_config.kind)
    if variant is not None:
        return repository.query_cases(variant)
    return repository.query_cases(
        QueryVariant.QWEN,
        qwen_queries=artifacts.rewrite(query),
    )


def rewrite_query(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
    *,
    query: ArtifactRef,
) -> None:
    query_config = config.query(query.name)
    tasks = BenchmarkRepository(config.run.benchmark_root).load_tasks()
    checkpoint = JsonlCheckpoint(artifacts.rewrite(query))
    pending = [task for task in tasks if task.task_id not in checkpoint.completed]
    if not pending:
        return

    client = ollama_client(config)
    prompt = PromptTemplate.from_file(query_config.prompt)
    model = ollama_model_info(config)
    provenance = {
        "rewrite_variant": query.name,
        "temperature": query_config.temperature,
        **model.provenance,
    }

    with SqliteCache(artifacts.cache) as cache:
        guard = thermal_guard(config)
        if query_config.kind == "agentic":
            answer_prompt = PromptTemplate.from_file(query_config.answer_prompt)
            compose_prompt = PromptTemplate.from_file(query_config.compose_prompt)
            rewriter = HistoryQueryAgent(
                client,
                model_name=model.identity,
                question_prompt=prompt,
                answer_prompt=answer_prompt,
                composition_prompt=compose_prompt,
                cache=cache,
                guard=guard,
                max_tokens=query_config.max_tokens,
                temperature=query_config.temperature,
            )
            prompt_revision = ":".join(
                (prompt.sha256, answer_prompt.sha256, compose_prompt.sha256)
            )
        else:
            rewriter = QueryRewriter(
                client,
                model_name=model.identity,
                prompt=prompt,
                cache=cache,
                guard=guard,
                max_tokens=query_config.max_tokens,
                temperature=query_config.temperature,
            )
            prompt_revision = prompt.sha256

        try:
            for index, task in enumerate(pending, start=1):
                details = {}
                if task.turn == 1:
                    rewritten = task.final_question
                    method = "identity"
                    revision = SINGLE_TURN_REWRITE_VERSION
                elif query_config.kind == "agentic":
                    outcome = rewriter.rewrite(task)
                    rewritten = outcome.query
                    method = "history_agent"
                    revision = prompt_revision
                    details = {
                        "resolution": outcome.resolution,
                        "clarification_questions": outcome.questions,
                        "resolution_status": outcome.status,
                        "evidence_ids": outcome.evidence_ids,
                    }
                    if outcome.composition is not None:
                        details["composition"] = outcome.composition
                else:
                    rewritten = rewriter.rewrite(task)
                    method = "qwen"
                    revision = prompt_revision

                checkpoint.append(
                    {
                        "task_id": task.task_id,
                        "query": rewritten,
                        "rewrite_method": method,
                        "rewrite_version": revision,
                        **details,
                        **provenance,
                    }
                )
                if index % 10 == 0 or index == len(pending):
                    progress(
                        f"rewrite {query.name}",
                        len(tasks) - len(pending) + index,
                        len(tasks),
                    )
        finally:
            client.unload()


def encode_bge_query(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
    *,
    query: ArtifactRef,
    features: ArtifactRef,
) -> None:
    repository = BenchmarkRepository(config.run.benchmark_root)
    cases = load_query_cases(repository, config, artifacts, query)
    encoded = _encode_query_cases(config, cases, artifacts.cache)
    BgeFeatureStore(artifacts.bge_features(features)).save(encoded)
    print(f"saved {len(encoded)} BGE query feature sets", flush=True)


def _encode_query_cases(
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
        cached = {text: cache.get(namespace, key) for text, key in keys.items()}
        features = {
            text: BgeFeatures(tuple(value["dense"]), value["sparse"])
            for text, value in cached.items()
            if value is not None
        }
        missing = [text for text, value in cached.items() if value is None]

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
                    encoded = dict(zip(batch, encoder.encode(batch), strict=True))
                    features.update(encoded)
                    cache.put_many(
                        namespace,
                        {
                            keys[text]: {
                                "dense": value.dense,
                                "sparse": value.sparse,
                            }
                            for text, value in encoded.items()
                        },
                    )
                    progress(
                        "bge encode",
                        min(start + len(batch), len(missing)),
                        len(missing),
                    )
    return {case.task_id: features[case.text] for case in cases}

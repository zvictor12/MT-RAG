from __future__ import annotations

import json
import os
import shutil

from mtrag.data import BenchmarkRepository
from mtrag.evaluation import (
    AlgorithmicGenerationEvaluator,
    BertScoreBatcher,
    summarize_generation_metrics,
)
from mtrag.experiments.artifacts import (
    JsonlCheckpoint,
    RunArtifacts,
    context_from_hit,
    context_record,
    read_jsonl,
    record_hits,
    task_record,
    write_jsonl_atomic,
)
from mtrag.experiments.common import ollama_client, progress, thermal_guard
from mtrag.experiments.spec import ExperimentConfig
from mtrag.llm import AnswerGenerator
from mtrag.llm.prompts import GENERATOR_PROMPT_VERSION
from mtrag.runtime import SqliteCache
from mtrag.runtime.state import write_json_atomic
from mtrag.schemas import BenchmarkTask, Context


def _generation_record(
    task: BenchmarkTask,
    contexts: list[Context],
    answer: str,
    *,
    config: ExperimentConfig,
    task_name: str,
) -> dict:
    record = task_record(task, include_targets=True)
    record["contexts"] = [
        context_record(context, rank)
        for rank, context in enumerate(contexts[:10], start=1)
    ]
    record["predictions"] = [{"text": answer}]
    record["pipeline"] = {
        "task": task_name,
        "model": config.models.ollama_model,
        "model_digest": config.models.ollama_digest,
        "prompt_version": GENERATOR_PROMPT_VERSION,
        "temperature": config.generation.temperature,
    }
    return record


def _generate(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
    *,
    task_name: str,
    contexts_by_task: dict[str, list[Context]],
    unload_after: bool,
) -> None:
    repository = BenchmarkRepository(config.run.benchmark_root)
    tasks = repository.load_tasks()
    output = artifacts.generation(task_name)
    checkpoint = JsonlCheckpoint(output)
    pending = [task for task in tasks if task.task_id not in checkpoint.completed]
    client = ollama_client(config)

    with SqliteCache(artifacts.cache) as cache:
        generator = AnswerGenerator(
            client,
            model_name=(
                f"{config.models.ollama_model}@{config.models.ollama_digest}"
            ),
            cache=cache,
            guard=thermal_guard(config),
            max_tokens=config.generation.max_tokens,
            temperature=config.generation.temperature,
        )
        try:
            for index, task in enumerate(pending, start=1):
                contexts = contexts_by_task.get(task.task_id, [])
                answer = generator.generate(task, contexts)
                checkpoint.append(
                    _generation_record(
                        task,
                        contexts,
                        answer,
                        config=config,
                        task_name=task_name,
                    )
                )
                if index % 10 == 0 or index == len(pending):
                    progress(
                        f"task {task_name}",
                        len(tasks) - len(pending) + index,
                        len(tasks),
                    )
        finally:
            if unload_after:
                client.unload()


def generate_task_b(config: ExperimentConfig, artifacts: RunArtifacts) -> None:
    tasks = BenchmarkRepository(config.run.benchmark_root).load_tasks()
    _generate(
        config,
        artifacts,
        task_name="b",
        contexts_by_task={task.task_id: list(task.contexts) for task in tasks},
        unload_after=False,
    )


def _generate_task_c(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
    *,
    candidate_name: str,
    task_name: str,
) -> None:
    records = read_jsonl(artifacts.candidates(candidate_name))
    contexts_by_task = {
        record["task_id"]: [
            context_from_hit(hit)
            for hit in record_hits(record)
            if hit.has_passage
        ][: config.generation.context_top_k]
        for record in records
    }
    _generate(
        config,
        artifacts,
        task_name=task_name,
        contexts_by_task=contexts_by_task,
        unload_after=True,
    )


def generate_task_c_bge(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
) -> None:
    _generate_task_c(
        config,
        artifacts,
        candidate_name="bge_selected",
        task_name="c_bge",
    )


def generate_task_c_bge_last(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
) -> None:
    _generate_task_c(
        config,
        artifacts,
        candidate_name="bge_dense_last",
        task_name="c_bge_last",
    )


def _copy_atomic(source, destination) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def _clone_task_jsonl(source, destination, task_name: str) -> None:
    records = read_jsonl(source)
    for record in records:
        pipeline = dict(record.get("pipeline", {}))
        pipeline["task"] = task_name
        record["pipeline"] = pipeline
    write_jsonl_atomic(destination, records)


def _reusable_task_c(decision: dict) -> str | None:
    return {
        "bge_dense_last": "c_bge_last",
        "bge_rrf_qwen_reranked": "c_bge",
    }.get(decision.get("winner"))


def generate_task_c_bge_selected(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
) -> None:
    decision = json.loads(artifacts.bge_winner.read_text(encoding="utf-8"))
    output = artifacts.generation("c_bge_selected")
    reusable = _reusable_task_c(decision)
    source = artifacts.generation(reusable) if reusable else None
    if source is not None and source.exists() and not output.exists():
        _clone_task_jsonl(source, output, "c_bge_selected")
        print("reused existing Task C for unchanged BGE winner", flush=True)
        return
    _generate_task_c(
        config,
        artifacts,
        candidate_name="bge_selected",
        task_name="c_bge_selected",
    )


def generate_task_c(config: ExperimentConfig, artifacts: RunArtifacts) -> None:
    _generate_task_c(
        config,
        artifacts,
        candidate_name="winner",
        task_name="c",
    )


def _evaluate_generation(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
    task_names: tuple[str, ...],
) -> None:
    pending = tuple(
        task_name
        for task_name in task_names
        if not (
            artifacts.generation_metrics(task_name).exists()
            and artifacts.generation_summary(task_name).exists()
        )
    )
    if not pending:
        return
    scorer = BertScoreBatcher(
        model_type=config.generation.bertscore_model,
        batch_size=config.generation.bertscore_batch_size,
        guard=thermal_guard(config),
    )
    evaluator = AlgorithmicGenerationEvaluator(scorer)
    for task_name in pending:
        records = read_jsonl(artifacts.generation(task_name))
        evaluated = evaluator.evaluate(records)
        write_jsonl_atomic(artifacts.generation_metrics(task_name), evaluated)
        summary = summarize_generation_metrics(evaluated)
        write_json_atomic(artifacts.generation_summary(task_name), summary)
        rb_agg = summary["metrics"].get("RB_agg", {}).get("mean")
        suffix = f", RB_agg={rb_agg:.4f}" if rb_agg is not None else ""
        print(
            f"evaluated Task {task_name.upper()}: {len(evaluated)}{suffix}",
            flush=True,
        )


def evaluate_generation_bge(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
) -> None:
    _evaluate_generation(config, artifacts, ("b", "c_bge"))


def evaluate_generation_bge_last(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
) -> None:
    _evaluate_generation(config, artifacts, ("c_bge_last",))


def evaluate_generation_bge_selected(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
) -> None:
    decision = json.loads(artifacts.bge_winner.read_text(encoding="utf-8"))
    reusable = _reusable_task_c(decision)
    source_metrics = artifacts.generation_metrics(reusable) if reusable else None
    source_summary = artifacts.generation_summary(reusable) if reusable else None
    output_metrics = artifacts.generation_metrics("c_bge_selected")
    output_summary = artifacts.generation_summary("c_bge_selected")
    if (
        source_metrics is not None
        and source_summary is not None
        and source_metrics.exists()
        and source_summary.exists()
        and not output_metrics.exists()
        and not output_summary.exists()
    ):
        _clone_task_jsonl(
            source_metrics,
            output_metrics,
            "c_bge_selected",
        )
        _copy_atomic(source_summary, output_summary)
        print("reused existing Task C metrics for unchanged BGE winner", flush=True)
    _evaluate_generation(config, artifacts, ("b", "c_bge_selected"))


def evaluate_generation(config: ExperimentConfig, artifacts: RunArtifacts) -> None:
    _evaluate_generation(config, artifacts, ("b", "c"))

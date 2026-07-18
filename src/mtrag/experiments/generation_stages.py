from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mtrag.data import BenchmarkRepository
from mtrag.data.jsonl import read_jsonl
from mtrag.evaluation import (
    AlgorithmicGenerationEvaluator,
    BertScoreBatcher,
    summarize_generation_metrics,
)
from mtrag.experiments.artifacts import (
    CandidateStore,
    JsonlCheckpoint,
    RunArtifacts,
)
from mtrag.experiments.common import (
    ollama_client,
    ollama_model_info,
    progress,
    thermal_guard,
)
from mtrag.experiments.spec import ExperimentConfig
from mtrag.llm import AnswerGenerator
from mtrag.llm.prompts import PromptTemplate
from mtrag.runtime import SqliteCache
from mtrag.runtime.state import write_json_atomic
from mtrag.schemas import ArtifactRef, BenchmarkTask, Context


def generate_job(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
    *,
    generation: ArtifactRef,
    context: ArtifactRef | None = None,
) -> None:
    job = config.generation_job(generation.name)
    generator_config = config.generator(job.generator)
    tasks = BenchmarkRepository(config.run.benchmark_root).load_tasks()
    output = artifacts.generation(generation)
    checkpoint = JsonlCheckpoint(output)
    pending = [task for task in tasks if task.task_id not in checkpoint.completed]
    if not pending:
        return

    if job.task == "b":
        contexts = {task.task_id: list(task.contexts) for task in tasks}
    else:
        contexts = CandidateStore(artifacts.candidates(context)).contexts(
            generator_config.context_top_k
        )
        missing = [task.task_id for task in tasks if not contexts.get(task.task_id)]
        if missing:
            raise RuntimeError(
                f"retrieval contexts are incomplete: "
                f"{len(tasks) - len(missing)}/{len(tasks)} tasks; "
                f"first missing task: {missing[0]}"
            )

    client = ollama_client(config)
    prompt = PromptTemplate.from_file(generator_config.prompt)
    model = ollama_model_info(config)
    pipeline = {
        "job": generation.name,
        "generator": job.generator,
        "prompt_sha256": prompt.sha256,
        "temperature": generator_config.temperature,
        **model.provenance,
    }
    with SqliteCache(artifacts.cache) as cache:
        generator = AnswerGenerator(
            client,
            model_name=model.identity,
            cache=cache,
            guard=thermal_guard(config),
            max_tokens=generator_config.max_tokens,
            temperature=generator_config.temperature,
            prompt=prompt,
        )
        try:
            for index, task in enumerate(pending, start=1):
                task_contexts = contexts.get(task.task_id, [])
                checkpoint.append(
                    _generation_record(
                        task,
                        task_contexts,
                        generator.generate(task, task_contexts),
                        pipeline,
                    )
                )
                if index % 10 == 0 or index == len(pending):
                    progress(
                        f"generate {generation.name}",
                        len(tasks) - len(pending) + index,
                        len(tasks),
                    )
        except BaseException:
            client.unload()
            raise


def unload_ollama(
    config: ExperimentConfig,
    _artifacts: RunArtifacts,
) -> None:
    ollama_client(config).unload()


def evaluate_generation_jobs(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
    *,
    jobs: Sequence[Mapping[str, Any]],
) -> None:
    """Evaluate several generation jobs while keeping one DeBERTa in memory."""
    evaluations = [GenerationEvaluation.load(artifacts, job) for job in jobs]
    unfinished = [evaluation for evaluation in evaluations if evaluation.pending]
    if unfinished:
        batch_size = config.evaluation.bertscore_batch_size
        scorer = BertScoreBatcher(
            model_type=config.evaluation.bertscore_model,
            batch_size=batch_size,
            chunk_size=batch_size * 8,
            guard=thermal_guard(config),
        )
        evaluator = AlgorithmicGenerationEvaluator(
            scorer,
            benchmark_root=config.run.benchmark_root,
        )
        for evaluation in unfinished:
            evaluator.evaluate_checkpointed(
                evaluation.records,
                evaluation.checkpoint,
            )

    for evaluation in evaluations:
        evaluation.write_summary()


@dataclass(slots=True)
class GenerationEvaluation:
    job_name: str
    records: list[dict[str, Any]]
    checkpoint: JsonlCheckpoint
    summary_path: Path

    @property
    def pending(self) -> bool:
        completed = self.checkpoint.completed
        return any(
            record["task_id"] not in completed
            for record in self.records
        )

    @classmethod
    def load(
        cls,
        artifacts: RunArtifacts,
        job: Mapping[str, Any],
    ) -> "GenerationEvaluation":
        generation = job["generation"]
        directory = artifacts.generation_evaluation(
            generation,
            job["evaluation_revision"],
        )
        return cls(
            job_name=generation.name,
            records=read_jsonl(artifacts.generation(generation)),
            checkpoint=JsonlCheckpoint(directory / "ibm-metrics.jsonl"),
            summary_path=directory / "ibm-summary.json",
        )

    def write_summary(self) -> None:
        evaluated = list(self.checkpoint.records.values())
        summary = summarize_generation_metrics(evaluated)
        write_json_atomic(self.summary_path, summary)
        rb_agg = summary["metrics"].get("RB_agg", {}).get("mean")
        suffix = f", RB_agg={rb_agg:.4f}" if rb_agg is not None else ""
        print(
            f"evaluated {self.job_name}: {len(evaluated)}{suffix}",
            flush=True,
        )


def _generation_record(
    task: BenchmarkTask,
    contexts: list[Context],
    answer: str,
    pipeline: Mapping[str, Any],
) -> dict:
    record = task.as_record(include_targets=True)
    serialized_contexts = []
    for rank, context in enumerate(contexts[:10], start=1):
        serialized = {
            "document_id": context.document_id,
            "text": context.text,
            "score": 1.0 / rank,
        }
        if context.score is not None:
            serialized["retriever_score"] = context.score
        if context.title is not None:
            serialized["title"] = context.title
        if context.url is not None:
            serialized["url"] = context.url
        serialized_contexts.append(serialized)
    record["contexts"] = serialized_contexts
    record["predictions"] = [{"text": answer}]
    record["pipeline"] = dict(pipeline)
    return record

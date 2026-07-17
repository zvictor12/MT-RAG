from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from mtrag.data import BenchmarkRepository
from mtrag.data.jsonl import read_jsonl
from mtrag.encoding import BgeFeatureStore
from mtrag.evaluation import RetrievalEvaluation, evaluate_retrieval
from mtrag.experiments.artifacts import (
    JsonlCheckpoint,
    RunArtifacts,
    materialize_prediction,
    ranking_record,
    record_hits,
)
from mtrag.experiments.common import chunks, progress, thermal_guard
from mtrag.experiments.query_stages import query_cases
from mtrag.experiments.spec import ExperimentConfig
from mtrag.reranking import BgeV2M3Scorer, RerankService
from mtrag.retrieval import DenseRetriever, ElserRetriever, SparseRetriever, rrf_fuse
from mtrag.retrieval.elasticsearch import ElasticsearchGateway
from mtrag.runtime import SqliteCache
from mtrag.runtime.state import write_json_atomic
from mtrag.schemas import SearchQuery


def retrieve(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
    *,
    reference: str,
    revision: str,
    query_name: str,
    query_revision: str,
    method: str,
    feature_revision: str | None = None,
) -> None:
    cases = query_cases(config, artifacts, query_name, query_revision)
    features = (
        BgeFeatureStore(
            artifacts.bge_features(query_name, feature_revision)
        ).load()
        if feature_revision is not None
        else None
    )
    retriever, top_k = _retriever(config, method)
    tasks = BenchmarkRepository(config.run.benchmark_root).tasks_by_id()
    checkpoint = JsonlCheckpoint(artifacts.candidates(reference, revision))
    pending = [case for case in cases if case.task_id not in checkpoint.completed]
    if not pending:
        return

    completed = len(cases) - len(pending)
    guard = thermal_guard(config)

    for batch in chunks(pending, config.retrieval.request_batch_size):
        guard.wait("cpu")
        queries = [
            SearchQuery(
                task_id=case.task_id,
                domain=case.domain,
                text=case.text,
                bge=features[case.task_id] if features is not None else None,
            )
            for case in batch
        ]
        results = retriever.search_many(queries, top_k=top_k)
        checkpoint.append_many(
            [
                ranking_record(tasks[case.task_id], results[case.task_id])
                for case in batch
            ]
        )
        completed += len(batch)
        progress(reference, completed, len(cases))


def fuse(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
    *,
    reference: str,
    revision: str,
    dense_reference: str,
    dense_revision: str,
    sparse_reference: str,
    sparse_revision: str,
) -> None:
    tasks = BenchmarkRepository(config.run.benchmark_root).tasks_by_id()
    dense = _records_by_id(artifacts.candidates(dense_reference, dense_revision))
    sparse = _records_by_id(artifacts.candidates(sparse_reference, sparse_revision))
    checkpoint = JsonlCheckpoint(artifacts.candidates(reference, revision))
    task_ids = [task_id for task_id in tasks if task_id in dense and task_id in sparse]
    pending = [task_id for task_id in task_ids if task_id not in checkpoint.completed]
    completed = len(task_ids) - len(pending)

    for batch in chunks(pending, 50):
        checkpoint.append_many(
            [
                ranking_record(
                    tasks[task_id],
                    rrf_fuse(
                        {
                            "dense": record_hits(dense[task_id]),
                            "sparse": record_hits(sparse[task_id]),
                        },
                        rank_constant=config.retrieval.rrf_rank_constant,
                        top_k=config.retrieval.rrf_top_k,
                    ),
                )
                for task_id in batch
            ]
        )
        completed += len(batch)
        progress(reference, completed, len(task_ids))


def rerank(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
    *,
    reference: str,
    revision: str,
    source_reference: str,
    source_revision: str,
    query_name: str,
    query_revision: str,
) -> None:
    cases = query_cases(config, artifacts, query_name, query_revision)
    queries = {case.task_id: case.text for case in cases}
    tasks = BenchmarkRepository(config.run.benchmark_root).tasks_by_id()
    source = _records_by_id(artifacts.candidates(source_reference, source_revision))
    checkpoint = JsonlCheckpoint(artifacts.candidates(reference, revision))
    pending = [
        task_id
        for task_id in queries
        if task_id in source and task_id not in checkpoint.completed
    ]
    if not pending:
        return

    completed = len(queries) - len(pending)
    scorer = BgeV2M3Scorer(
        config.models.reranker_path,
        batch_size=config.models.reranker_batch_size,
        max_length=config.models.reranker_max_length,
        guard=thermal_guard(config),
    )
    try:
        with SqliteCache(artifacts.cache) as cache:
            service = RerankService(
                scorer,
                cache=cache,
                model_revision=config.models.reranker_revision,
                max_length=config.models.reranker_max_length,
            )
            for batch in chunks(pending, config.reranking.task_batch_size):
                reranked = service.rerank_many(
                    {task_id: queries[task_id] for task_id in batch},
                    {
                        task_id: record_hits(source[task_id])[
                            : config.reranking.input_top_k
                        ]
                        for task_id in batch
                    },
                    top_k=config.reranking.output_top_k,
                )
                checkpoint.append_many(
                    [
                        ranking_record(tasks[task_id], reranked[task_id])
                        for task_id in batch
                    ]
                )
                completed += len(batch)
                progress(reference, completed, len(queries))
    finally:
        scorer.close()


def evaluate_task_a(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
    *,
    reference: str,
    revision: str,
    evaluation_revision: str,
) -> RetrievalEvaluation:
    tasks = BenchmarkRepository(config.run.benchmark_root).load_tasks()
    prediction = artifacts.prediction(reference, revision)
    materialize_prediction(
        artifacts.candidates(reference, revision),
        prediction,
        top_k=config.retrieval.prediction_top_k,
        tasks=tasks,
    )
    evaluation = evaluate_retrieval(config.run.benchmark_root, prediction)
    write_json_atomic(
        artifacts.retrieval_report(reference, revision, evaluation_revision),
        asdict(evaluation),
    )
    print(
        f"{reference}: nDCG@5={evaluation.metrics.ndcg[5]:.4f}, "
        f"Recall@5={evaluation.metrics.recall[5]:.4f}",
        flush=True,
    )
    return evaluation


def _retriever(config: ExperimentConfig, method: str):
    gateway = ElasticsearchGateway(
        config.services.elasticsearch_url,
        request_batch_size=config.retrieval.request_batch_size,
    )
    if method == "dense":
        return (
            DenseRetriever(
                gateway,
                candidate_multiplier=config.retrieval.dense_candidate_multiplier,
                rescore_oversample=config.retrieval.dense_rescore_oversample,
            ),
            config.retrieval.dense_top_k,
        )
    if method == "sparse":
        return SparseRetriever(gateway), config.retrieval.sparse_top_k
    return ElserRetriever(gateway), config.retrieval.elser_top_k


def _records_by_id(path: Path) -> dict[str, dict]:
    records = read_jsonl(path)
    indexed = {record["task_id"]: record for record in records}
    if len(indexed) != len(records):
        raise ValueError(f"duplicate task_id in {path}")
    return indexed

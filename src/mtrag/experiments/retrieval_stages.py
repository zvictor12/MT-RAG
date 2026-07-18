from __future__ import annotations

from dataclasses import asdict
from itertools import batched

from mtrag.data import BenchmarkRepository
from mtrag.encoding import BgeFeatureStore
from mtrag.evaluation import RetrievalEvaluation, evaluate_retrieval
from mtrag.experiments.artifacts import (
    CandidateStore,
    RunArtifacts,
)
from mtrag.experiments.common import progress, thermal_guard
from mtrag.experiments.query_stages import load_query_cases
from mtrag.experiments.spec import ExperimentConfig
from mtrag.reranking import BgeV2M3Scorer, RerankService
from mtrag.retrieval import DenseRetriever, ElserRetriever, SparseRetriever, rrf_fuse
from mtrag.retrieval.elasticsearch import ElasticsearchGateway
from mtrag.runtime import SqliteCache
from mtrag.runtime.state import write_json_atomic
from mtrag.schemas import ArtifactRef, SearchQuery


def retrieve(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
    *,
    output: ArtifactRef,
    query: ArtifactRef,
    method: str,
    features: ArtifactRef | None = None,
) -> None:
    repository = BenchmarkRepository(config.run.benchmark_root)
    cases = load_query_cases(repository, config, artifacts, query)
    encoded = (
        BgeFeatureStore(artifacts.bge_features(features)).load()
        if features is not None
        else None
    )
    gateway = ElasticsearchGateway(
        config.services.elasticsearch_url,
        request_batch_size=config.retrieval.request_batch_size,
    )
    if method == "dense":
        retriever = DenseRetriever(
            gateway,
            candidate_multiplier=config.retrieval.dense_candidate_multiplier,
            rescore_oversample=config.retrieval.dense_rescore_oversample,
        )
        top_k = config.retrieval.dense_top_k
    elif method == "sparse":
        retriever = SparseRetriever(gateway)
        top_k = config.retrieval.sparse_top_k
    else:
        retriever = ElserRetriever(gateway)
        top_k = config.retrieval.elser_top_k

    tasks = repository.tasks_by_id()
    store = CandidateStore(artifacts.candidates(output))
    pending = [case for case in cases if case.task_id not in store.completed]
    if not pending:
        return

    completed = len(cases) - len(pending)
    guard = thermal_guard(config)

    for batch in batched(pending, config.retrieval.request_batch_size):
        guard.wait("cpu")
        queries = [
            SearchQuery(
                task_id=case.task_id,
                domain=case.domain,
                text=case.text,
                bge=encoded[case.task_id] if encoded is not None else None,
            )
            for case in batch
        ]
        results = retriever.search_many(queries, top_k=top_k)
        store.append_hits(
            tasks,
            {case.task_id: results[case.task_id] for case in batch},
        )
        completed += len(batch)
        progress(output.name, completed, len(cases))


def fuse(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
    *,
    output: ArtifactRef,
    dense: ArtifactRef,
    sparse: ArtifactRef,
) -> None:
    tasks = BenchmarkRepository(config.run.benchmark_root).tasks_by_id()
    dense_results = CandidateStore(artifacts.candidates(dense)).rankings(tasks)
    sparse_results = CandidateStore(artifacts.candidates(sparse)).rankings(tasks)
    store = CandidateStore(artifacts.candidates(output))
    task_ids = [
        task_id
        for task_id in tasks
        if task_id in dense_results and task_id in sparse_results
    ]
    pending = [task_id for task_id in task_ids if task_id not in store.completed]
    completed = len(task_ids) - len(pending)

    for batch in batched(pending, 50):
        fused = {
            task_id: rrf_fuse(
                {
                    "dense": dense_results[task_id].hits,
                    "sparse": sparse_results[task_id].hits,
                },
                rank_constant=config.retrieval.rrf_rank_constant,
                top_k=config.retrieval.rrf_top_k,
            )
            for task_id in batch
        }
        store.append_hits(tasks, fused)
        completed += len(batch)
        progress(output.name, completed, len(task_ids))


def rerank(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
    *,
    output: ArtifactRef,
    source: ArtifactRef,
    query: ArtifactRef,
) -> None:
    repository = BenchmarkRepository(config.run.benchmark_root)
    cases = load_query_cases(repository, config, artifacts, query)
    queries = {case.task_id: case.text for case in cases}
    tasks = repository.tasks_by_id()
    source_results = CandidateStore(artifacts.candidates(source)).rankings(tasks)
    store = CandidateStore(artifacts.candidates(output))
    pending = [
        task_id
        for task_id in queries
        if task_id in source_results and task_id not in store.completed
    ]
    if not pending:
        return

    completed = len(queries) - len(pending)
    with BgeV2M3Scorer(
        config.models.reranker_path,
        batch_size=config.models.reranker_batch_size,
        max_length=config.models.reranker_max_length,
        guard=thermal_guard(config),
    ) as scorer, SqliteCache(artifacts.cache) as cache:
        service = RerankService(
            scorer,
            cache=cache,
            model_revision=config.models.reranker_revision,
            max_length=config.models.reranker_max_length,
        )
        for batch in batched(pending, config.reranking.task_batch_size):
            reranked = service.rerank_many(
                {task_id: queries[task_id] for task_id in batch},
                {
                    task_id: source_results[task_id].hits[
                        : config.reranking.input_top_k
                    ]
                    for task_id in batch
                },
                top_k=config.reranking.output_top_k,
            )
            store.append_hits(tasks, reranked)
            completed += len(batch)
            progress(output.name, completed, len(queries))


def evaluate_task_a(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
    *,
    output: ArtifactRef,
    evaluation_revision: str,
) -> RetrievalEvaluation:
    tasks = BenchmarkRepository(config.run.benchmark_root).load_tasks()
    prediction = artifacts.prediction(output)
    CandidateStore(artifacts.candidates(output)).write_prediction(
        prediction,
        top_k=config.retrieval.prediction_top_k,
        tasks=tasks,
    )
    evaluation = evaluate_retrieval(config.run.benchmark_root, prediction)
    write_json_atomic(
        artifacts.retrieval_report(output, evaluation_revision),
        asdict(evaluation),
    )
    print(
        f"{output.name}: nDCG@5={evaluation.metrics.ndcg[5]:.4f}, "
        f"Recall@5={evaluation.metrics.recall[5]:.4f}",
        flush=True,
    )
    return evaluation

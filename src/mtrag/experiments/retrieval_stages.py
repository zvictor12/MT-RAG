from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

from mtrag.data import BenchmarkRepository
from mtrag.encoding import BgeFeatureStore
from mtrag.evaluation import (
    RetrievalEvaluation,
    evaluate_retrieval,
    load_benchmark_qrels,
    load_rankings_jsonl,
    paired_bootstrap,
)
from mtrag.experiments.artifacts import (
    JsonlCheckpoint,
    RunArtifacts,
    materialize_prediction,
    ranking_record,
    read_jsonl,
    record_hits,
)
from mtrag.experiments.common import chunks, progress, thermal_guard
from mtrag.experiments.query_stages import feature_key
from mtrag.experiments.spec import ExperimentConfig
from mtrag.reranking import BgeV2M3Scorer, RerankService
from mtrag.retrieval import DenseRetriever, ElserRetriever, SparseRetriever, rrf_fuse
from mtrag.retrieval.elasticsearch import ElasticsearchGateway
from mtrag.runtime import SqliteCache
from mtrag.runtime.state import write_json_atomic
from mtrag.schemas import BgeFeatures, QueryCase, QueryVariant, SearchHit, SearchQuery


BGE_BASE_RUNS = (
    "bge_dense_last",
    "bge_dense_qwen",
    "bge_dense_gold",
    "bge_sparse_qwen",
    "bge_rrf_qwen",
)

BGE_VARIANT_PIPELINES = {
    "last": {
        "dense": "bge_dense_last",
        "sparse": "bge_sparse_last",
        "rrf": "bge_rrf_last",
        "reranked": "bge_rrf_last_reranked",
    },
    "qwen_t0": {
        "dense": "bge_dense_qwen",
        "sparse": "bge_sparse_qwen",
        "rrf": "bge_rrf_qwen",
        "reranked": "bge_rrf_qwen_reranked",
    },
    "qwen_t02": {
        "dense": "bge_dense_qwen_t02",
        "sparse": "bge_sparse_qwen_t02",
        "rrf": "bge_rrf_qwen_t02",
        "reranked": "bge_rrf_qwen_t02_reranked",
    },
}

BGE_VARIANT_BASE_RUNS = (
    "bge_sparse_last",
    "bge_rrf_last",
    "bge_dense_qwen_t02",
    "bge_sparse_qwen_t02",
    "bge_rrf_qwen_t02",
)

BGE_VARIANT_RERANK_RUNS = (
    "bge_rrf_last_reranked",
    "bge_rrf_qwen_t02_reranked",
)

DENSE_REWRITE_RUNS = {
    "last": "bge_dense_last",
    "qwen_t0": "bge_dense_qwen",
    "qwen_t02": "bge_dense_qwen_t02",
}


def _gateway(config: ExperimentConfig) -> ElasticsearchGateway:
    return ElasticsearchGateway(
        config.services.elasticsearch_url,
        request_batch_size=config.retrieval.request_batch_size,
    )


def _queries(
    cases: Sequence[QueryCase],
    features: Mapping[str, BgeFeatures] | None = None,
) -> list[SearchQuery]:
    return [
        SearchQuery(
            task_id=case.task_id,
            domain=case.domain,
            text=case.text,
            bge=features[feature_key(case)] if features is not None else None,
        )
        for case in cases
    ]


def _search_checkpoint(
    *,
    label: str,
    cases: Sequence[QueryCase],
    queries: Sequence[SearchQuery],
    tasks_by_id,
    retriever,
    top_k: int,
    request_batch_size: int,
    path: Path,
    guard,
) -> None:
    checkpoint = JsonlCheckpoint(path)
    pairs = [
        (case, query)
        for case, query in zip(cases, queries, strict=True)
        if case.task_id not in checkpoint.completed
    ]
    completed_before = len(cases) - len(pairs)
    for batch in chunks(pairs, request_batch_size):
        guard.wait("cpu")
        results = retriever.search_many(
            [query for _case, query in batch],
            top_k=top_k,
        )
        checkpoint.append_many(
            [
                ranking_record(tasks_by_id[case.task_id], results[case.task_id])
                for case, _query in batch
            ]
        )
        completed_before += len(batch)
        progress(label, completed_before, len(cases))


def retrieve_bge(config: ExperimentConfig, artifacts: RunArtifacts) -> None:
    repository = BenchmarkRepository(config.run.benchmark_root)
    tasks_by_id = repository.tasks_by_id()
    features = BgeFeatureStore(artifacts.bge_features).load()
    gateway = _gateway(config)
    dense = DenseRetriever(
        gateway,
        candidate_multiplier=config.retrieval.dense_candidate_multiplier,
        rescore_oversample=config.retrieval.dense_rescore_oversample,
    )
    sparse = SparseRetriever(gateway)
    guard = thermal_guard(config)

    cases_by_variant = {
        QueryVariant.LAST: repository.query_cases(QueryVariant.LAST),
        QueryVariant.QWEN: repository.query_cases(
            QueryVariant.QWEN,
            qwen_queries=artifacts.qwen_queries,
        ),
        QueryVariant.GOLD: repository.query_cases(QueryVariant.GOLD),
    }
    for variant in (QueryVariant.LAST, QueryVariant.QWEN, QueryVariant.GOLD):
        cases = cases_by_variant[variant]
        _search_checkpoint(
            label=f"dense {variant.value}",
            cases=cases,
            queries=_queries(cases, features),
            tasks_by_id=tasks_by_id,
            retriever=dense,
            top_k=config.retrieval.dense_top_k,
            request_batch_size=config.retrieval.request_batch_size,
            path=artifacts.candidates(f"bge_dense_{variant.value}"),
            guard=guard,
        )

    qwen_cases = cases_by_variant[QueryVariant.QWEN]
    _search_checkpoint(
        label="sparse qwen",
        cases=qwen_cases,
        queries=_queries(qwen_cases, features),
        tasks_by_id=tasks_by_id,
        retriever=sparse,
        top_k=config.retrieval.sparse_top_k,
        request_batch_size=config.retrieval.request_batch_size,
        path=artifacts.candidates("bge_sparse_qwen"),
        guard=guard,
    )


def _generated_cases(
    repository: BenchmarkRepository,
    artifacts: RunArtifacts,
    variant: QueryVariant,
) -> tuple[QueryCase, ...]:
    return repository.query_cases(
        variant,
        qwen_queries=artifacts.rewrite_queries(variant.value),
    )


def _variant_cases(
    repository: BenchmarkRepository,
    artifacts: RunArtifacts,
    variant: str,
) -> tuple[QueryCase, ...]:
    selected = QueryVariant(variant)
    if selected is QueryVariant.LAST:
        return repository.query_cases(selected)
    return _generated_cases(repository, artifacts, selected)


def retrieve_bge_variants(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
) -> None:
    repository = BenchmarkRepository(config.run.benchmark_root)
    tasks_by_id = repository.tasks_by_id()
    features = BgeFeatureStore(artifacts.bge_features).load()
    gateway = _gateway(config)
    dense = DenseRetriever(
        gateway,
        candidate_multiplier=config.retrieval.dense_candidate_multiplier,
        rescore_oversample=config.retrieval.dense_rescore_oversample,
    )
    sparse = SparseRetriever(gateway)
    guard = thermal_guard(config)

    last_cases = repository.query_cases(QueryVariant.LAST)
    _search_checkpoint(
        label="sparse last",
        cases=last_cases,
        queries=_queries(last_cases, features),
        tasks_by_id=tasks_by_id,
        retriever=sparse,
        top_k=config.retrieval.sparse_top_k,
        request_batch_size=config.retrieval.request_batch_size,
        path=artifacts.candidates("bge_sparse_last"),
        guard=guard,
    )

    t02_cases = _generated_cases(
        repository,
        artifacts,
        QueryVariant.QWEN_T02,
    )
    for method, retriever, top_k in (
        ("dense", dense, config.retrieval.dense_top_k),
        ("sparse", sparse, config.retrieval.sparse_top_k),
    ):
        _search_checkpoint(
            label=f"{method} qwen_t02",
            cases=t02_cases,
            queries=_queries(t02_cases, features),
            tasks_by_id=tasks_by_id,
            retriever=retriever,
            top_k=top_k,
            request_batch_size=config.retrieval.request_batch_size,
            path=artifacts.candidates(f"bge_{method}_qwen_t02"),
            guard=guard,
        )


def retrieve_elser(config: ExperimentConfig, artifacts: RunArtifacts) -> None:
    repository = BenchmarkRepository(config.run.benchmark_root)
    tasks_by_id = repository.tasks_by_id()
    decision = json.loads(
        artifacts.rewrite_winner.read_text(encoding="utf-8")
    )
    variant = str(decision["winner"])
    cases = _variant_cases(repository, artifacts, variant)
    _search_checkpoint(
        label=f"elser {variant}",
        cases=cases,
        queries=_queries(cases),
        tasks_by_id=tasks_by_id,
        retriever=ElserRetriever(_gateway(config)),
        top_k=config.retrieval.elser_top_k,
        request_batch_size=config.retrieval.request_batch_size,
        path=artifacts.candidates("elser_selected"),
        guard=thermal_guard(config),
    )


def _records_by_id(path: Path) -> dict[str, dict]:
    records = read_jsonl(path)
    indexed = {record["task_id"]: record for record in records}
    if len(indexed) != len(records):
        raise ValueError(f"duplicate task_id in {path}")
    return indexed


def fuse_bge(config: ExperimentConfig, artifacts: RunArtifacts) -> None:
    _fuse_bge_variant(config, artifacts, "qwen")


def _fuse_bge_variant(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
    suffix: str,
) -> None:
    repository = BenchmarkRepository(config.run.benchmark_root)
    tasks_by_id = repository.tasks_by_id()
    dense = _records_by_id(artifacts.candidates(f"bge_dense_{suffix}"))
    sparse = _records_by_id(artifacts.candidates(f"bge_sparse_{suffix}"))
    checkpoint = JsonlCheckpoint(artifacts.candidates(f"bge_rrf_{suffix}"))

    task_ids = [
        task_id
        for task_id in tasks_by_id
        if task_id in dense and task_id in sparse
    ]
    pending = [
        task_id for task_id in task_ids if task_id not in checkpoint.completed
    ]
    completed = len(task_ids) - len(pending)
    for batch in chunks(pending, 50):
        records = []
        for task_id in batch:
            hits = rrf_fuse(
                {
                    "dense": record_hits(dense[task_id]),
                    "sparse": record_hits(sparse[task_id]),
                },
                rank_constant=config.retrieval.rrf_rank_constant,
                top_k=config.retrieval.rrf_top_k,
            )
            records.append(ranking_record(tasks_by_id[task_id], hits))
        checkpoint.append_many(records)
        completed += len(batch)
        progress(f"bge rrf {suffix}", completed, len(task_ids))


def fuse_bge_variants(config: ExperimentConfig, artifacts: RunArtifacts) -> None:
    for suffix in ("last", "qwen_t02"):
        _fuse_bge_variant(config, artifacts, suffix)


def _evaluate_candidate(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
    name: str,
) -> RetrievalEvaluation:
    tasks = BenchmarkRepository(config.run.benchmark_root).load_tasks()
    materialize_prediction(
        artifacts.candidates(name),
        artifacts.prediction(name),
        top_k=config.retrieval.prediction_top_k,
        tasks=tasks,
    )
    return evaluate_retrieval(
        load_benchmark_qrels(config.run.benchmark_root),
        load_rankings_jsonl(artifacts.prediction(name)),
    )


def _save_evaluation(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
    name: str,
) -> RetrievalEvaluation:
    evaluation = _evaluate_candidate(config, artifacts, name)
    write_json_atomic(artifacts.retrieval_report(name), asdict(evaluation))
    print(
        f"{name}: nDCG@5={evaluation.metrics.ndcg[5]:.4f}, "
        f"Recall@5={evaluation.metrics.recall[5]:.4f}",
        flush=True,
    )
    return evaluation


def evaluate_bge_base(config: ExperimentConfig, artifacts: RunArtifacts) -> None:
    for name in BGE_BASE_RUNS:
        _save_evaluation(config, artifacts, name)


def evaluate_bge_variants_base(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
) -> None:
    for name in BGE_VARIANT_BASE_RUNS:
        _save_evaluation(config, artifacts, name)


def evaluate_elser_base(config: ExperimentConfig, artifacts: RunArtifacts) -> None:
    _save_evaluation(config, artifacts, "elser_selected")


def _rerank(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
    *,
    source_name: str,
    output_name: str,
    variant: QueryVariant = QueryVariant.QWEN,
    rewrite_path: Path | None = None,
) -> None:
    repository = BenchmarkRepository(config.run.benchmark_root)
    cases = (
        repository.query_cases(variant)
        if variant is QueryVariant.LAST
        else repository.query_cases(
            variant,
            qwen_queries=rewrite_path or artifacts.qwen_queries,
        )
    )
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
            _rerank_with_service(
                config,
                artifacts,
                cases=cases,
                source_name=source_name,
                output_name=output_name,
                service=service,
            )
    finally:
        scorer.close()


def _rerank_with_service(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
    *,
    cases: Sequence[QueryCase],
    source_name: str,
    output_name: str,
    service: RerankService,
) -> None:
    queries = {case.task_id: case.text for case in cases}
    tasks_by_id = BenchmarkRepository(config.run.benchmark_root).tasks_by_id()
    source = _records_by_id(artifacts.candidates(source_name))
    checkpoint = JsonlCheckpoint(artifacts.candidates(output_name))
    pending = [
        task_id
        for task_id in queries
        if task_id in source and task_id not in checkpoint.completed
    ]
    completed = len(queries) - len(pending)
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
                ranking_record(tasks_by_id[task_id], reranked[task_id])
                for task_id in batch
            ]
        )
        completed += len(batch)
        progress(output_name, completed, len(queries))


def rerank_bge(config: ExperimentConfig, artifacts: RunArtifacts) -> None:
    _rerank(
        config,
        artifacts,
        source_name="bge_rrf_qwen",
        output_name="bge_rrf_qwen_reranked",
    )


def evaluate_bge_rerank(config: ExperimentConfig, artifacts: RunArtifacts) -> None:
    _save_evaluation(config, artifacts, "bge_rrf_qwen_reranked")


def rerank_bge_variants(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
) -> None:
    repository = BenchmarkRepository(config.run.benchmark_root)
    jobs = (
        (
            repository.query_cases(QueryVariant.LAST),
            "bge_rrf_last",
            "bge_rrf_last_reranked",
        ),
        (
            _generated_cases(
                repository,
                artifacts,
                QueryVariant.QWEN_T02,
            ),
            "bge_rrf_qwen_t02",
            "bge_rrf_qwen_t02_reranked",
        ),
    )
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
            for cases, source_name, output_name in jobs:
                _rerank_with_service(
                    config,
                    artifacts,
                    cases=cases,
                    source_name=source_name,
                    output_name=output_name,
                    service=service,
                )
    finally:
        scorer.close()


def evaluate_bge_variants_rerank(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
) -> None:
    for name in BGE_VARIANT_RERANK_RUNS:
        _save_evaluation(config, artifacts, name)


def decide_reranker(config: ExperimentConfig, artifacts: RunArtifacts) -> None:
    decision = _reranker_decision(
        config,
        artifacts,
        baseline_name="bge_rrf_qwen",
        candidate_name="bge_rrf_qwen_reranked",
    )
    write_json_atomic(artifacts.reranker_gate, decision)
    print(
        f"reranker gate: {'enabled' if decision['enabled'] else 'disabled'}",
        flush=True,
    )


def _reranker_decision(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
    *,
    baseline_name: str,
    candidate_name: str,
) -> dict:
    baseline = _evaluate_candidate(config, artifacts, baseline_name)
    reranked = _evaluate_candidate(config, artifacts, candidate_name)
    comparison = paired_bootstrap(
        baseline,
        reranked,
        metric="ndcg",
        cutoff=5,
        samples=config.reranking.bootstrap_samples,
        seed=config.reranking.bootstrap_seed,
    )
    gain = reranked.metrics.ndcg[5] - baseline.metrics.ndcg[5]
    enabled = (
        gain >= config.reranking.minimum_ndcg5_gain
        and comparison.probability_improvement
        >= config.reranking.minimum_improvement_probability
    )
    return {
        "enabled": enabled,
        "baseline": baseline_name,
        "candidate": candidate_name,
        "ndcg5_gain": gain,
        "minimum_ndcg5_gain": config.reranking.minimum_ndcg5_gain,
        "probability_improvement": comparison.probability_improvement,
        "minimum_improvement_probability": (
            config.reranking.minimum_improvement_probability
        ),
        "paired_bootstrap": asdict(comparison),
    }


def decide_bge_variants(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
) -> None:
    variants = {
        variant: _reranker_decision(
            config,
            artifacts,
            baseline_name=names["rrf"],
            candidate_name=names["reranked"],
        )
        for variant, names in BGE_VARIANT_PIPELINES.items()
    }
    write_json_atomic(
        artifacts.reranker_variants,
        {
            "enabled": any(item["enabled"] for item in variants.values()),
            "variants": variants,
        },
    )
    for variant, decision in variants.items():
        print(
            f"reranker {variant}: "
            f"{'enabled' if decision['enabled'] else 'disabled'}, "
            f"gain={decision['ndcg5_gain']:+.4f}",
            flush=True,
        )


def reranker_enabled(artifacts: RunArtifacts) -> bool:
    if artifacts.reranker_variants.exists():
        decision = json.loads(
            artifacts.reranker_variants.read_text(encoding="utf-8")
        )
        return bool(decision["enabled"])
    decision = json.loads(artifacts.reranker_gate.read_text(encoding="utf-8"))
    return bool(decision["enabled"])


def rerank_elser(config: ExperimentConfig, artifacts: RunArtifacts) -> None:
    decision = json.loads(
        artifacts.rewrite_winner.read_text(encoding="utf-8")
    )
    variant = QueryVariant(str(decision["winner"]))
    _rerank(
        config,
        artifacts,
        source_name="elser_selected",
        output_name="elser_selected_reranked",
        variant=variant,
        rewrite_path=(
            None
            if variant is QueryVariant.LAST
            else artifacts.rewrite_queries(variant.value)
        ),
    )


def evaluate_elser_rerank(config: ExperimentConfig, artifacts: RunArtifacts) -> None:
    _save_evaluation(config, artifacts, "elser_selected_reranked")


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def select_bge(config: ExperimentConfig, artifacts: RunArtifacts) -> None:
    use_reranker = reranker_enabled(artifacts)
    rrf_name = "bge_rrf_qwen_reranked" if use_reranker else "bge_rrf_qwen"
    candidate_names = (
        "bge_dense_qwen",
        "bge_sparse_qwen",
        rrf_name,
    )
    scores = {
        name: _evaluate_candidate(config, artifacts, name).metrics.ndcg[5]
        for name in candidate_names
    }
    selected = min(scores, key=lambda name: (-scores[name], name))
    _copy_atomic(
        artifacts.candidates(selected),
        artifacts.candidates("bge_selected"),
    )
    _copy_atomic(
        artifacts.prediction(selected),
        artifacts.prediction("bge_selected"),
    )
    write_json_atomic(
        artifacts.bge_winner,
        {
            "winner": selected,
            "metric": "ndcg@5",
            "score": scores[selected],
            "scores": scores,
            "reranker_enabled": use_reranker,
            "candidate_artifact": str(artifacts.candidates("bge_selected")),
            "task_a_prediction": str(artifacts.prediction("bge_selected")),
        },
    )
    print(f"BGE winner: {selected} ({scores[selected]:.4f})", flush=True)


def _report_ndcg5(artifacts: RunArtifacts, name: str) -> float:
    report = json.loads(
        artifacts.retrieval_report(name).read_text(encoding="utf-8")
    )
    return float(report["metrics"]["ndcg"]["5"])


def select_rewrite_variant(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
) -> None:
    del config
    scores = {
        variant: _report_ndcg5(artifacts, run_name)
        for variant, run_name in DENSE_REWRITE_RUNS.items()
    }
    selected = min(scores, key=lambda name: (-scores[name], name))
    write_json_atomic(
        artifacts.rewrite_winner,
        {
            "winner": selected,
            "metric": "ndcg@5",
            "score": scores[selected],
            "scores": scores,
            "retrieval_method": "bge_dense",
        },
    )
    print(
        f"rewrite winner: {selected} ({scores[selected]:.4f})",
        flush=True,
    )


def select_bge_variants(
    config: ExperimentConfig,
    artifacts: RunArtifacts,
) -> None:
    del config
    reranker = json.loads(
        artifacts.reranker_variants.read_text(encoding="utf-8")
    )["variants"]
    candidates: dict[str, str] = {}
    for variant, names in BGE_VARIANT_PIPELINES.items():
        for method in ("dense", "sparse", "rrf"):
            candidates[names[method]] = variant
        if reranker[variant]["enabled"]:
            candidates[names["reranked"]] = variant

    scores = {name: _report_ndcg5(artifacts, name) for name in candidates}
    selected = min(scores, key=lambda name: (-scores[name], name))
    _copy_atomic(
        artifacts.candidates(selected),
        artifacts.candidates("bge_selected"),
    )
    _copy_atomic(
        artifacts.prediction(selected),
        artifacts.prediction("bge_selected"),
    )
    write_json_atomic(
        artifacts.bge_winner,
        {
            "winner": selected,
            "query_variant": candidates[selected],
            "metric": "ndcg@5",
            "score": scores[selected],
            "scores": scores,
            "reranker": reranker,
            "candidate_artifact": str(artifacts.candidates("bge_selected")),
            "task_a_prediction": str(artifacts.prediction("bge_selected")),
        },
    )
    print(
        f"BGE winner: {selected} ({scores[selected]:.4f})",
        flush=True,
    )


def select_winner(config: ExperimentConfig, artifacts: RunArtifacts) -> None:
    candidate_names = ["bge_selected", "elser_selected"]
    if (
        reranker_enabled(artifacts)
        and artifacts.candidates("elser_selected_reranked").exists()
    ):
        candidate_names.append("elser_selected_reranked")
    scores = {
        name: _evaluate_candidate(config, artifacts, name).metrics.ndcg[5]
        for name in candidate_names
    }
    winner = min(scores, key=lambda name: (-scores[name], name))
    _copy_atomic(artifacts.candidates(winner), artifacts.candidates("winner"))
    _copy_atomic(artifacts.prediction(winner), artifacts.prediction("winner"))
    write_json_atomic(
        artifacts.winner,
        {
            "winner": winner,
            "metric": "ndcg@5",
            "scores": scores,
            "rewrite_variant": json.loads(
                artifacts.rewrite_winner.read_text(encoding="utf-8")
            )["winner"],
            "candidate_artifact": str(artifacts.candidates("winner")),
            "task_a_prediction": str(artifacts.prediction("winner")),
        },
    )
    print(f"winner: {winner} ({scores[winner]:.4f})", flush=True)

from __future__ import annotations

from mtrag.experiments.artifacts import RunArtifacts
from mtrag.experiments.generation_stages import (
    evaluate_generation,
    evaluate_generation_bge,
    evaluate_generation_bge_last,
    evaluate_generation_bge_selected,
    generate_task_b,
    generate_task_c,
    generate_task_c_bge,
    generate_task_c_bge_last,
    generate_task_c_bge_selected,
)
from mtrag.experiments.query_stages import (
    encode_bge,
    encode_bge_variants,
    rewrite_qwen,
    rewrite_qwen_t02,
)
from mtrag.experiments.retrieval_stages import (
    decide_bge_variants,
    decide_reranker,
    evaluate_bge_base,
    evaluate_bge_rerank,
    evaluate_bge_variants_base,
    evaluate_bge_variants_rerank,
    evaluate_elser_base,
    evaluate_elser_rerank,
    fuse_bge,
    fuse_bge_variants,
    rerank_bge,
    rerank_bge_variants,
    rerank_elser,
    retrieve_bge,
    retrieve_bge_variants,
    retrieve_elser,
    select_bge,
    select_bge_variants,
    select_rewrite_variant,
    select_winner,
)
from mtrag.experiments.spec import ExperimentConfig
from mtrag.experiments.preflight import preflight


STAGES = {
    "preflight": preflight,
    "rewrite_qwen": rewrite_qwen,
    "rewrite_qwen_t02": rewrite_qwen_t02,
    "encode_bge": encode_bge,
    "encode_bge_variants": encode_bge_variants,
    "retrieve_elser": retrieve_elser,
    "retrieve_bge": retrieve_bge,
    "retrieve_bge_variants": retrieve_bge_variants,
    "fuse_bge": fuse_bge,
    "fuse_bge_variants": fuse_bge_variants,
    "evaluate_bge_base": evaluate_bge_base,
    "evaluate_bge_variants_base": evaluate_bge_variants_base,
    "evaluate_elser_base": evaluate_elser_base,
    "rerank_bge": rerank_bge,
    "rerank_bge_variants": rerank_bge_variants,
    "evaluate_bge_rerank": evaluate_bge_rerank,
    "evaluate_bge_variants_rerank": evaluate_bge_variants_rerank,
    "decide_reranker": decide_reranker,
    "decide_bge_variants": decide_bge_variants,
    "select_bge": select_bge,
    "select_bge_variants": select_bge_variants,
    "select_rewrite_variant": select_rewrite_variant,
    "rerank_elser": rerank_elser,
    "evaluate_elser_rerank": evaluate_elser_rerank,
    "select_winner": select_winner,
    "generate_task_b": generate_task_b,
    "generate_task_c_bge": generate_task_c_bge,
    "evaluate_generation_bge": evaluate_generation_bge,
    "generate_task_c_bge_last": generate_task_c_bge_last,
    "evaluate_generation_bge_last": evaluate_generation_bge_last,
    "generate_task_c_bge_selected": generate_task_c_bge_selected,
    "evaluate_generation_bge_selected": evaluate_generation_bge_selected,
    "generate_task_c": generate_task_c,
    "evaluate_generation": evaluate_generation,
}


def run_stage(
    name: str,
    config: ExperimentConfig,
    artifacts: RunArtifacts,
) -> None:
    try:
        stage = STAGES[name]
    except KeyError as error:
        raise ValueError(f"unknown experiment stage: {name}") from error
    stage(config, artifacts)

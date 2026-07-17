"""Thin adapters around the official IBM MT-RAG evaluators."""

from .generation import (
    AlgorithmicGenerationEvaluator,
    BertScoreBatcher,
    summarize_generation_metrics,
)
from .retrieval import (
    DEFAULT_CUTOFFS,
    DomainEvaluation,
    MetricValues,
    RetrievalEvaluation,
    evaluate_retrieval,
)
from .writer import (
    MAX_OFFICIAL_CONTEXTS,
    deterministic_rank_score,
    make_retrieval_record,
    write_retrieval_jsonl,
)

__all__ = [
    "AlgorithmicGenerationEvaluator",
    "BertScoreBatcher",
    "DEFAULT_CUTOFFS",
    "DomainEvaluation",
    "MAX_OFFICIAL_CONTEXTS",
    "MetricValues",
    "RetrievalEvaluation",
    "deterministic_rank_score",
    "evaluate_retrieval",
    "make_retrieval_record",
    "summarize_generation_metrics",
    "write_retrieval_jsonl",
]

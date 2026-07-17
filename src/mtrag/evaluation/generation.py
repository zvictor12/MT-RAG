from __future__ import annotations

import hashlib
import json
import logging
import math
import tempfile
import types
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path
from statistics import fmean
from typing import Any, Protocol

from mtrag.interfaces import BatchGuard, NoopGuard

from .ibm import load_ibm_module


LOGGER = logging.getLogger(__name__)
IBM_BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"


class SemanticScorer(Protocol):
    def score(
        self,
        candidates: Sequence[str],
        references: Sequence[str],
    ) -> tuple[list[float], list[float], list[float]]: ...


class EvaluationCheckpoint(Protocol):
    @property
    def completed(self) -> set[str]: ...

    def append_many(self, records: Sequence[Mapping[str, Any]]) -> None: ...


class BertScoreBatcher:
    """Cache DeBERTa once and score many pairs in real GPU batches."""

    def __init__(
        self,
        *,
        model_type: str = IBM_BERTSCORE_MODEL,
        device: str = "cuda:0",
        batch_size: int = 4,
        chunk_size: int | None = None,
        guard: BatchGuard | None = None,
    ) -> None:
        from bert_score import BERTScorer

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if chunk_size is None:
            chunk_size = batch_size * 8
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.model_type = model_type
        self.scorer = BERTScorer(
            model_type=model_type,
            lang="en",
            device=device,
            batch_size=batch_size,
            rescale_with_baseline=True,
        )
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.guard = guard or NoopGuard()

    def score(
        self,
        candidates: Sequence[str],
        references: Sequence[str],
    ) -> tuple[list[float], list[float], list[float]]:
        precision: list[float] = []
        recall: list[float] = []
        f1: list[float] = []
        for start in range(0, len(candidates), self.chunk_size):
            self.guard.wait("gpu")
            end = min(start + self.chunk_size, len(candidates))
            p_values, r_values, f_values = self.scorer.score(
                list(candidates[start:end]),
                list(references[start:end]),
                batch_size=self.batch_size,
            )
            precision.extend(float(value) for value in p_values)
            recall.extend(float(value) for value in r_values)
            f1.extend(float(value) for value in f_values)
            LOGGER.info("BERTScore: %d/%d semantic pairs", end, len(candidates))
        return precision, recall, f1


class AlgorithmicGenerationEvaluator:
    """Run IBM's Task B/C evaluator with precomputed batched BERTScore."""

    def __init__(
        self,
        semantic_scorer: SemanticScorer,
        *,
        benchmark_root: str | Path | None = None,
    ) -> None:
        root = (
            Path(benchmark_root).expanduser().resolve()
            if benchmark_root is not None
            else _default_benchmark_root()
        )
        self.script = root / "scripts" / "evaluation" / "run_algorithmic.py"
        self.config = root / "scripts" / "evaluation" / "config.yaml"
        self.module = _load_official_module(root)
        self.semantic_scorer = semantic_scorer
        self.source_digest = _source_digest(self.script, self.config)

    def evaluate(self, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if not records:
            return []

        pairs = _semantic_pairs(records)
        candidates = [prediction for prediction, _reference in pairs]
        references = [reference for _prediction, reference in pairs]
        precision, recall, f1 = self.semantic_scorer.score(candidates, references)
        bertscore = _BertScoreLookup(
            pairs,
            precision,
            recall,
            f1,
            model_type=getattr(self.semantic_scorer, "model_type", None),
        )

        previous_bertscore = self.module.bertscore_metric
        previous_rouge = self.module.rouge_evaluator
        self.module.bertscore_metric = bertscore
        self.module.rouge_evaluator = _RougeScoreMetric()
        try:
            return self._run_official(records)
        finally:
            self.module.bertscore_metric = previous_bertscore
            self.module.rouge_evaluator = previous_rouge

    def evaluate_checkpointed(
        self,
        records: Sequence[Mapping[str, Any]],
        checkpoint: EvaluationCheckpoint,
        *,
        record_batch_size: int = 32,
    ) -> int:
        """Evaluate unfinished records and durably append each completed batch."""
        if record_batch_size <= 0:
            raise ValueError("record_batch_size must be positive")
        completed = checkpoint.completed
        pending = [
            record
            for record in records
            if _task_id(record) not in completed
        ]
        for start in range(0, len(pending), record_batch_size):
            evaluated = self.evaluate(pending[start : start + record_batch_size])
            checkpoint.append_many(evaluated)
        return len(pending)

    def _run_official(
        self,
        records: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        with tempfile.TemporaryDirectory(prefix="mtrag-ibm-eval-") as directory:
            root = Path(directory)
            input_path = root / "input.jsonl"
            output_path = root / "output.jsonl"
            _write_jsonl(input_path, records)
            self.module.run_algorithmic_judges(
                str(self.config),
                str(input_path),
                str(output_path),
            )
            evaluated = _read_jsonl(output_path)
            expected_ids = [_task_id(record) for record in records]
            actual_ids = [_task_id(record) for record in evaluated]
            if actual_ids != expected_ids:
                raise RuntimeError(
                    "IBM evaluator changed the task sequence: "
                    f"expected {expected_ids!r}, got {actual_ids!r}"
                )
            return evaluated


class _DeferredMetric:
    def compute(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("official metric backend was not installed")


class _BertScoreLookup:
    def __init__(
        self,
        pairs: Sequence[tuple[str, str]],
        precision: Sequence[float],
        recall: Sequence[float],
        f1: Sequence[float],
        *,
        model_type: str | None,
    ) -> None:
        self.values = {
            pair: (float(p), float(r), float(f))
            for pair, p, r, f in zip(pairs, precision, recall, f1, strict=True)
        }
        self.model_type = model_type

    def compute(
        self,
        *,
        predictions: Sequence[str],
        references: Sequence[str],
        model_type: str | None = None,
        lang: str | None = None,
        rescale_with_baseline: bool | None = None,
        **kwargs: Any,
    ) -> dict[str, list[float]]:
        if kwargs:
            raise ValueError(
                "unsupported BERTScore arguments from the IBM evaluator: "
                f"{sorted(kwargs)}"
            )
        if lang != "en":
            raise ValueError(
                f"IBM evaluator requires English BERTScore, got {lang!r}"
            )
        if rescale_with_baseline is not True:
            raise ValueError(
                "IBM evaluator must request BERTScore baseline rescaling"
            )
        expected_model = self.model_type or IBM_BERTSCORE_MODEL
        if model_type != expected_model:
            raise ValueError(
                f"official evaluator requires BERTScore model {model_type!r}, "
                f"but the batch scorer loaded {expected_model!r}"
            )
        scores = [
            self.values[(prediction, reference)]
            for prediction, reference in zip(predictions, references, strict=True)
        ]
        return {
            "precision": [score[0] for score in scores],
            "recall": [score[1] for score in scores],
            "f1": [score[2] for score in scores],
        }


class _RougeScoreMetric:
    def compute(
        self,
        *,
        predictions: Sequence[str],
        references: Sequence[str],
        rouge_types: Sequence[str],
        use_aggregator: bool,
        use_stemmer: bool,
        **_kwargs: Any,
    ) -> dict[str, list[float]]:
        if use_aggregator:
            raise ValueError("IBM's algorithmic evaluator requests raw ROUGE scores")
        from rouge_score.rouge_scorer import RougeScorer

        scorer = RougeScorer(list(rouge_types), use_stemmer=use_stemmer)
        output = {name: [] for name in rouge_types}
        for prediction, reference in zip(predictions, references, strict=True):
            scores = scorer.score(reference, prediction)
            for name in rouge_types:
                output[name].append(float(scores[name].fmeasure))
        return output


@cache
def _load_official_module(benchmark_root: Path) -> types.ModuleType:
    placeholder = types.ModuleType("evaluate")
    placeholder.load = lambda *_args, **_kwargs: _DeferredMetric()  # type: ignore[attr-defined]
    return load_ibm_module(
        benchmark_root,
        "run_algorithmic.py",
        module_overrides={"evaluate": placeholder},
    )


def _default_benchmark_root() -> Path:
    project_root = Path(__file__).resolve().parents[3]
    return project_root.parent / "mt-rag-benchmark"


def _source_digest(script: Path, config: Path) -> str:
    if not config.is_file():
        raise FileNotFoundError(f"IBM evaluator config not found: {config}")
    digest = hashlib.sha256()
    for path in (script, config):
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _semantic_pairs(
    records: Sequence[Mapping[str, Any]],
) -> list[tuple[str, str]]:
    pairs: dict[tuple[str, str], None] = {}
    for record in records:
        prediction = _prediction(record)
        for item in (*record.get("targets", []), *record.get("contexts", [])):
            pairs[(prediction, item["text"])] = None
    return list(pairs)


def _prediction(record: Mapping[str, Any]) -> str:
    predictions = record.get("predictions") or []
    if not predictions or not isinstance(predictions[0].get("text"), str):
        raise ValueError(f"Missing prediction for task {record.get('task_id')}")
    return predictions[0]["text"]


def _task_id(record: Mapping[str, Any]) -> str:
    task_id = record.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("generation record requires a non-empty task_id")
    return task_id


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for original in records:
            record = dict(original)
            record.pop("metrics", None)
            stream.write(json.dumps(record, ensure_ascii=False))
            stream.write("\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def summarize_generation_metrics(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Average each official metric per task, then across tasks."""
    values: dict[str, list[float]] = {}
    for record in records:
        for name, raw_values in (record.get("metrics") or {}).items():
            if not isinstance(raw_values, Sequence) or isinstance(
                raw_values,
                (str, bytes),
            ):
                continue
            task_values = [
                float(value)
                for value in raw_values
                if isinstance(value, (int, float)) and math.isfinite(value)
            ]
            if task_values:
                values.setdefault(str(name), []).append(fmean(task_values))
    return {
        "task_count": len(records),
        "metrics": {
            name: {
                "mean": fmean(metric_values),
                "task_count": len(metric_values),
            }
            for name, metric_values in sorted(values.items())
        },
    }

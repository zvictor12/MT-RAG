import html
import logging
import math
import re
import string
from collections import Counter
from collections.abc import Mapping, Sequence
from statistics import fmean
from typing import Any, Callable, Protocol

from mtrag.interfaces import BatchGuard, NoopGuard


LOGGER = logging.getLogger(__name__)


class SemanticScorer(Protocol):
    def score(
        self,
        candidates: Sequence[str],
        references: Sequence[str],
    ) -> tuple[list[float], list[float], list[float]]: ...


def normalize_text(text: str) -> str:
    lowered = text.lower()
    without_punctuation = "".join(
        character
        for character in lowered
        if character not in string.punctuation
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def token_recall(prediction: str, target: str) -> float:
    target_tokens = normalize_text(target).split()
    if not target_tokens:
        return 0.0
    common = Counter(normalize_text(prediction).split()) & Counter(target_tokens)
    return sum(common.values()) / len(target_tokens)


def harmonic(values: Sequence[float]) -> float:
    if not values or any(value <= 0 for value in values):
        return 0.0
    return len(values) / sum(1.0 / value for value in values)


class BertScoreBatcher:
    """Cache DeBERTa once and score many pairs in real GPU batches."""

    def __init__(
        self,
        *,
        model_type: str = "microsoft/deberta-xlarge-mnli",
        device: str = "cuda:0",
        batch_size: int = 4,
        chunk_size: int = 512,
        guard: BatchGuard | None = None,
    ) -> None:
        from bert_score import BERTScorer

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
            LOGGER.info(
                "BERTScore: %d/%d semantic pairs",
                end,
                len(candidates),
            )
        return precision, recall, f1


class AlgorithmicGenerationEvaluator:
    """Batched equivalent of IBM's algorithmic Task B/C metrics."""

    def __init__(
        self,
        semantic_scorer: SemanticScorer,
        *,
        rouge_l: Callable[[str, str], float] | None = None,
    ) -> None:
        self.semantic_scorer = semantic_scorer
        if rouge_l is None:
            from rouge_score.rouge_scorer import RougeScorer

            scorer = RougeScorer(["rougeL"], use_stemmer=False)
            rouge_l = lambda prediction, target: scorer.score(
                target,
                prediction,
            )["rougeL"].fmeasure
        self.rouge_l = rouge_l

    def evaluate(self, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        candidates: list[str] = []
        references: list[str] = []
        locations: list[tuple[int, str]] = []

        for row_index, record in enumerate(records):
            prediction = _prediction(record)
            for target in record.get("targets", []):
                candidates.append(prediction)
                references.append(target["text"])
                locations.append((row_index, "target"))
            for context in record.get("contexts", []):
                candidates.append(prediction)
                references.append(context["text"])
                locations.append((row_index, "passage"))

        precision, recall, _f1 = self.semantic_scorer.score(candidates, references)
        semantic: list[dict[str, list[float]]] = [
            {"target_p": [], "target_r": [], "passage_p": []}
            for _ in records
        ]
        for index, (row_index, kind) in enumerate(locations):
            if kind == "target":
                semantic[row_index]["target_p"].append(precision[index])
                semantic[row_index]["target_r"].append(recall[index])
            else:
                semantic[row_index]["passage_p"].append(precision[index])

        output = []
        for row_index, original in enumerate(records):
            record = dict(original)
            prediction = _prediction(record)
            targets = [target["text"] for target in record.get("targets", [])]
            passages = [context["text"] for context in record.get("contexts", [])]
            values = semantic[row_index]
            rouge_targets = [self._rouge_l(prediction, target) for target in targets]
            extractive = [self._extractiveness(prediction, text) for text in passages]
            metrics = dict(record.get("metrics") or {})
            metrics.update(
                {
                    "Recall": [token_recall(prediction, text) for text in targets],
                    "RougeL_stemFalse": rouge_targets,
                    "BertscoreP": values["target_p"],
                    "BertscoreR": values["target_r"],
                    "BertKPrec": values["passage_p"],
                    "Extractiveness_RougeL": extractive,
                    "Length": [len(prediction) for _ in targets],
                }
            )
            bert_recall = values["target_r"][0] if values["target_r"] else -1.0
            rouge = rouge_targets[0] if rouge_targets else 0.0
            passage_precision = max(values["passage_p"], default=-1.0)
            metrics["RB_agg"] = [
                harmonic(
                    (
                        (bert_recall + 1.0) / 2.0,
                        rouge,
                        (passage_precision + 1.0) / 2.0,
                    )
                )
            ]
            record["metrics"] = metrics
            output.append(record)
        return output

    def _rouge_l(self, prediction: str, target: str) -> float:
        return self.rouge_l(prediction, target)

    def _extractiveness(self, prediction: str, passage: str) -> float:
        clean_prediction = _clean_for_extractiveness(prediction)
        clean_passage = _clean_for_extractiveness(passage)
        if "".join(clean_passage.split()) in "".join(clean_prediction.split()):
            return 1.0
        return self._rouge_l(clean_prediction, clean_passage)


def _prediction(record: Mapping[str, Any]) -> str:
    predictions = record.get("predictions") or []
    if not predictions or not isinstance(predictions[0].get("text"), str):
        raise ValueError(f"Missing prediction for task {record.get('task_id')}")
    return predictions[0]["text"]


def _clean_for_extractiveness(text: str) -> str:
    # Preserve the official evaluator's operation order.  In particular it
    # removes punctuation before parsing HTML, even though parsing first would
    # be more natural.  Matching that order keeps already published scores
    # comparable for passages containing markup.
    stripped = re.sub(r"[^\w\s]", "", text.replace("\n", " ")).lower()
    decoded = html.unescape(stripped)
    try:
        from bs4 import BeautifulSoup

        plain = BeautifulSoup(decoded, features="lxml").get_text()
    except ImportError:
        plain = re.sub(r"<[^>]+>", "", decoded)
    return plain


def summarize_generation_metrics(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Average each metric per task, then across tasks with that metric."""
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

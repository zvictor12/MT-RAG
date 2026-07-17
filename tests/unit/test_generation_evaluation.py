import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from mtrag.evaluation.generation import (
    AlgorithmicGenerationEvaluator,
    BertScoreBatcher,
    _BertScoreLookup,
    summarize_generation_metrics,
)


UPSTREAM_FIXTURE = '''
import evaluate
import json

bertscore_metric = evaluate.load("bertscore")
rouge_evaluator = evaluate.load("rouge")


def run_algorithmic_judges(_config, input_file, output_file):
    with open(input_file, encoding="utf-8") as source, open(
        output_file, "w", encoding="utf-8"
    ) as destination:
        for line in source:
            record = json.loads(line)
            prediction = record["predictions"][0]["text"]
            target = record["targets"][0]["text"]
            passage = record["contexts"][0]["text"]
            target_scores = bertscore_metric.compute(
                predictions=[prediction],
                references=[target],
                model_type="official-model",
                lang="en",
                rescale_with_baseline=True,
            )
            passage_scores = bertscore_metric.compute(
                predictions=[prediction],
                references=[passage],
                model_type="official-model",
                lang="en",
                rescale_with_baseline=True,
            )
            rouge = rouge_evaluator.compute(
                predictions=[prediction],
                references=[target],
                rouge_types=["rougeL"],
                use_aggregator=False,
                use_stemmer=False,
            )
            record["metrics"] = {
                "OfficialSentinel": [7.0],
                "BertscoreP": target_scores["precision"],
                "BertscoreR": target_scores["recall"],
                "BertKPrec": passage_scores["precision"],
                "RougeL_stemFalse": rouge["rougeL"],
            }
            destination.write(json.dumps(record) + "\\n")
'''


def record(task_id: str) -> dict:
    return {
        "task_id": task_id,
        "predictions": [{"text": "the answer"}],
        "targets": [{"text": "answer"}],
        "contexts": [{"text": "context one"}, {"text": "context two"}],
    }


class FakeSemanticScorer:
    model_type = "official-model"

    def __init__(self) -> None:
        self.calls: list[list[tuple[str, str]]] = []

    def score(self, candidates, references):
        self.calls.append(list(zip(candidates, references, strict=True)))
        count = len(references)
        return [0.4] * count, [0.5] * count, [0.6] * count


class FakeCheckpoint:
    def __init__(self, completed=()) -> None:
        self.records = {task_id: record(task_id) for task_id in completed}
        self.batches: list[list[dict]] = []

    @property
    def completed(self) -> set[str]:
        return set(self.records)

    def append_many(self, records) -> None:
        batch = [dict(item) for item in records]
        self.batches.append(batch)
        self.records.update((item["task_id"], item) for item in batch)


class GenerationEvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.benchmark_root = Path(self.temporary.name)
        evaluation = self.benchmark_root / "scripts" / "evaluation"
        evaluation.mkdir(parents=True)
        (evaluation / "run_algorithmic.py").write_text(
            UPSTREAM_FIXTURE,
            encoding="utf-8",
        )
        (evaluation / "config.yaml").write_text("fixture: true\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def evaluator(self, scorer=None) -> AlgorithmicGenerationEvaluator:
        return AlgorithmicGenerationEvaluator(
            scorer or FakeSemanticScorer(),
            benchmark_root=self.benchmark_root,
        )

    def test_executes_upstream_runner_with_batched_semantic_scores(self) -> None:
        scorer = FakeSemanticScorer()
        output = self.evaluator(scorer).evaluate([record("q")])

        self.assertEqual(len(scorer.calls), 1)
        self.assertEqual(len(scorer.calls[0]), 3)
        self.assertEqual(output[0]["metrics"]["OfficialSentinel"], [7.0])
        self.assertEqual(output[0]["metrics"]["BertscoreR"], [0.5])
        self.assertEqual(output[0]["metrics"]["BertKPrec"], [0.4])

    def test_upstream_import_does_not_load_or_replace_evaluate_package(self) -> None:
        sentinel = types.ModuleType("evaluate")
        previous = sys.modules.get("evaluate")
        sys.modules["evaluate"] = sentinel
        try:
            self.evaluator()
            self.assertIs(sys.modules["evaluate"], sentinel)
        finally:
            if previous is None:
                sys.modules.pop("evaluate", None)
            else:
                sys.modules["evaluate"] = previous

    def test_checkpoint_skips_completed_records_and_appends_batches(self) -> None:
        scorer = FakeSemanticScorer()
        checkpoint = FakeCheckpoint(completed=("q1",))

        count = self.evaluator(scorer).evaluate_checkpointed(
            [record("q1"), record("q2"), record("q3")],
            checkpoint,
            record_batch_size=1,
        )

        self.assertEqual(count, 2)
        self.assertEqual(len(scorer.calls), 2)
        self.assertEqual(len(checkpoint.batches), 2)
        self.assertEqual(checkpoint.completed, {"q1", "q2", "q3"})

    def test_rejects_changed_task_order_from_upstream(self) -> None:
        evaluator = self.evaluator()
        original = evaluator.module.run_algorithmic_judges

        def reversed_output(config, input_file, output_file):
            original(config, input_file, output_file)
            path = Path(output_file)
            path.write_text(
                "".join(reversed(path.read_text().splitlines(keepends=True))),
                encoding="utf-8",
            )

        evaluator.module.run_algorithmic_judges = reversed_output
        try:
            with self.assertRaisesRegex(RuntimeError, "task sequence"):
                evaluator.evaluate([record("q1"), record("q2")])
        finally:
            evaluator.module.run_algorithmic_judges = original

    def test_rejects_incompatible_official_bertscore_arguments(self) -> None:
        lookup = _BertScoreLookup(
            [("prediction", "reference")],
            [0.1],
            [0.2],
            [0.3],
            model_type="official-model",
        )
        valid = {
            "predictions": ["prediction"],
            "references": ["reference"],
            "model_type": "official-model",
            "lang": "en",
            "rescale_with_baseline": True,
        }

        with self.assertRaisesRegex(ValueError, "English"):
            lookup.compute(**(valid | {"lang": "fr"}))
        with self.assertRaisesRegex(ValueError, "baseline"):
            lookup.compute(**(valid | {"rescale_with_baseline": False}))
        with self.assertRaisesRegex(ValueError, "batch scorer loaded"):
            lookup.compute(**(valid | {"model_type": "other-model"}))
        with self.assertRaisesRegex(ValueError, "unsupported"):
            lookup.compute(**valid, idf=True)

    def test_default_thermal_chunk_is_eight_microbatches(self) -> None:
        class FakeBertScorer:
            calls: list[int] = []

            def __init__(self, **_kwargs) -> None:
                pass

            def score(self, candidates, _references, **_kwargs):
                self.calls.append(len(candidates))
                values = [0.5] * len(candidates)
                return values, values, values

        class Guard:
            def __init__(self) -> None:
                self.calls = 0

            def wait(self, _resource: str) -> None:
                self.calls += 1

        module = types.ModuleType("bert_score")
        module.BERTScorer = FakeBertScorer
        guard = Guard()
        with patch.dict(sys.modules, {"bert_score": module}):
            scorer = BertScoreBatcher(batch_size=2, guard=guard)
            scorer.score(["p"] * 17, ["r"] * 17)

        self.assertEqual(FakeBertScorer.calls, [16, 1])
        self.assertEqual(guard.calls, 2)

    def test_source_digest_includes_official_config(self) -> None:
        initial = self.evaluator().source_digest
        config = self.benchmark_root / "scripts" / "evaluation" / "config.yaml"
        config.write_text("fixture: changed\n", encoding="utf-8")

        self.assertNotEqual(initial, self.evaluator().source_digest)

    def test_summary_weights_tasks_instead_of_passage_count(self) -> None:
        summary = summarize_generation_metrics(
            [
                {"metrics": {"BertKPrec": [0.0, 1.0], "RB_agg": [0.2]}},
                {"metrics": {"BertKPrec": [1.0], "RB_agg": [0.6]}},
                {"metrics": {"BertKPrec": [], "RB_agg": [0.4]}},
            ]
        )
        self.assertEqual(summary["task_count"], 3)
        self.assertEqual(summary["metrics"]["BertKPrec"]["mean"], 0.75)
        self.assertEqual(summary["metrics"]["BertKPrec"]["task_count"], 2)
        self.assertAlmostEqual(summary["metrics"]["RB_agg"]["mean"], 0.4)


if __name__ == "__main__":
    unittest.main()

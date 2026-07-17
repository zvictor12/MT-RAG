import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mtrag.experiments.artifacts import JsonlCheckpoint, RunArtifacts
from mtrag.experiments.generation_stages import evaluate_generation_jobs


class FakeEvaluator:
    instances = 0

    def __init__(self, _scorer, *, benchmark_root) -> None:
        self.benchmark_root = benchmark_root
        type(self).instances += 1

    def evaluate_checkpointed(self, records, checkpoint) -> int:
        evaluated = [
            dict(record, metrics={"RB_agg": [0.4]})
            for record in records
            if record["task_id"] not in checkpoint.completed
        ]
        checkpoint.append_many(evaluated)
        return len(evaluated)


class GenerationStagesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.artifacts = RunArtifacts(Path(self.temporary.name))
        self.config = SimpleNamespace(
            evaluation=SimpleNamespace(
                bertscore_model="official-model",
                bertscore_batch_size=3,
            ),
            run=SimpleNamespace(benchmark_root=Path("benchmark")),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def job(self, name: str, generation: str, evaluation: str) -> dict[str, str]:
        return {
            "job_name": name,
            "generation_revision": generation,
            "evaluation_revision": evaluation,
        }

    def write_generation(self, name: str, revision: str, task_id: str) -> dict:
        record = {"task_id": task_id, "predictions": [{"text": "answer"}]}
        path = self.artifacts.generation(name, revision)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        return record

    @patch("mtrag.experiments.generation_stages.thermal_guard", return_value=object())
    @patch("mtrag.experiments.generation_stages.BertScoreBatcher")
    @patch("mtrag.experiments.generation_stages.AlgorithmicGenerationEvaluator")
    def test_reuses_one_scorer_for_multiple_jobs(
        self,
        evaluator_type,
        scorer_type,
        _guard,
    ) -> None:
        FakeEvaluator.instances = 0
        evaluator_type.side_effect = FakeEvaluator
        scorer_type.return_value = object()
        first = self.job("first", "a" * 64, "b" * 64)
        second = self.job("second", "c" * 64, "d" * 64)
        self.write_generation("first", "a" * 64, "q1")
        self.write_generation("second", "c" * 64, "q2")

        evaluate_generation_jobs(
            self.config,
            self.artifacts,
            jobs=(first, second),
        )

        scorer_type.assert_called_once_with(
            model_type="official-model",
            batch_size=3,
            chunk_size=24,
            guard=_guard.return_value,
        )
        self.assertEqual(FakeEvaluator.instances, 1)
        for job in (first, second):
            summary = self.artifacts.generation_summary(
                job["job_name"],
                job["generation_revision"],
                job["evaluation_revision"],
            )
            self.assertEqual(json.loads(summary.read_text())["task_count"], 1)

    @patch(
        "mtrag.experiments.generation_stages.BertScoreBatcher",
        side_effect=AssertionError("resume must not load BERTScore"),
    )
    def test_complete_checkpoint_writes_summary_without_loading_scorer(
        self,
        scorer_type,
    ) -> None:
        job = self.job("complete", "e" * 64, "f" * 64)
        record = self.write_generation("complete", "e" * 64, "q1")
        checkpoint = JsonlCheckpoint(
            self.artifacts.generation_metrics(
                "complete",
                "e" * 64,
                "f" * 64,
            )
        )
        checkpoint.append(dict(record, metrics={"RB_agg": [0.7]}))

        evaluate_generation_jobs(self.config, self.artifacts, jobs=(job,))

        scorer_type.assert_not_called()
        summary = self.artifacts.generation_summary(
            "complete",
            "e" * 64,
            "f" * 64,
        )
        self.assertEqual(
            json.loads(summary.read_text())["metrics"]["RB_agg"]["mean"],
            0.7,
        )


if __name__ == "__main__":
    unittest.main()

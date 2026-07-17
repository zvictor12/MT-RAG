import json
import tempfile
import unittest
from pathlib import Path

from mtrag.experiments.artifacts import (
    JsonlCheckpoint,
    RunArtifacts,
    lock_run_definition,
    materialize_prediction,
    ranking_record,
    read_jsonl,
    record_hits,
    write_jsonl_atomic,
)
from mtrag.schemas import BenchmarkTask, Message, SearchHit


class JsonlCheckpointTest(unittest.TestCase):
    def test_resume_repairs_only_an_interrupted_final_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "items.jsonl"
            path.write_bytes(b'{"task_id":"one","value":1}\n{"task_id"')

            checkpoint = JsonlCheckpoint(path)
            self.assertEqual(checkpoint.completed, {"one"})
            checkpoint.append({"task_id": "two", "value": 2})

            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([record["task_id"] for record in records], ["one", "two"])

    def test_corruption_before_the_last_record_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "items.jsonl"
            path.write_text('{"task_id"\n{"task_id":"two"}\n')
            with self.assertRaisesRegex(ValueError, "corrupt JSONL"):
                JsonlCheckpoint(path)

    def test_valid_final_record_without_newline_gets_a_separator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "items.jsonl"
            path.write_text('{"task_id":"one"}')

            checkpoint = JsonlCheckpoint(path)
            checkpoint.append({"task_id": "two"})

            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([record["task_id"] for record in records], ["one", "two"])

    def test_append_many_validates_before_writing_the_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "items.jsonl"
            checkpoint = JsonlCheckpoint(path)
            checkpoint.append_many(
                ({"task_id": "one"}, {"task_id": "two"})
            )
            before = path.read_bytes()

            with self.assertRaisesRegex(ValueError, "duplicate checkpoint"):
                checkpoint.append_many(
                    ({"task_id": "three"}, {"task_id": "three"})
                )

            self.assertEqual(path.read_bytes(), before)


class ArtifactMappingTest(unittest.TestCase):
    def test_ranking_round_trip_keeps_raw_score_and_rank(self) -> None:
        task = BenchmarkTask(
            task_id="q<::>1",
            conversation_id="q",
            turn=1,
            collection="collection",
            domain="clapnq",
            messages=(Message("user", "question"),),
        )
        hit = SearchHit(
            document_id="doc",
            score=12.5,
            rank=1,
            source="dense",
            title="Title",
            text="Passage",
        )

        record = ranking_record(task, [hit])
        restored = record_hits(record)

        self.assertEqual(record["contexts"][0]["score"], 1.0)
        self.assertEqual(restored, [hit])

    def test_run_artifacts_separates_candidates_and_official_predictions(self) -> None:
        paths = RunArtifacts(Path("run"))
        self.assertEqual(paths.candidates("dense"), Path("run/candidates/dense.jsonl"))
        self.assertEqual(
            paths.prediction("dense"),
            Path("run/predictions/task_a/dense.jsonl"),
        )
        self.assertEqual(
            paths.bge_winner,
            Path("run/decisions/bge-winner.json"),
        )
        self.assertEqual(
            paths.generation("c_bge"),
            Path("run/predictions/task_c_bge.jsonl"),
        )

    def test_run_definition_cannot_change_during_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "definition.json"
            lock_run_definition(path, {"config": "one", "prompt": "v1"})
            lock_run_definition(path, {"config": "one", "prompt": "v1"})

            with self.assertRaisesRegex(RuntimeError, "new --run-dir"):
                lock_run_definition(path, {"config": "two", "prompt": "v1"})

    def test_official_prediction_fills_tasks_without_a_query_variant(self) -> None:
        tasks = [
            BenchmarkTask(
                task_id=f"q<::>{turn}",
                conversation_id="q",
                turn=turn,
                collection="collection",
                domain="clapnq",
                messages=(Message("user", "question"),),
            )
            for turn in (1, 2)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / "candidates.jsonl"
            prediction = root / "prediction.jsonl"
            hit = SearchHit(
                "doc",
                12.0,
                1,
                "dense",
                title="Title",
                text="Long internal passage",
            )
            write_jsonl_atomic(
                candidates,
                [ranking_record(tasks[0], [hit])],
            )

            count = materialize_prediction(
                candidates,
                prediction,
                top_k=10,
                tasks=tasks,
            )

            records = read_jsonl(prediction)
            self.assertEqual(count, 2)
            self.assertEqual([row["task_id"] for row in records], ["q<::>1", "q<::>2"])
            self.assertEqual(
                records[0]["contexts"],
                [{"document_id": "doc", "score": 1.0}],
            )
            self.assertEqual(records[1]["contexts"], [])


if __name__ == "__main__":
    unittest.main()

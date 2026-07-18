import json
import tempfile
import unittest
from pathlib import Path

from mtrag.data.jsonl import read_jsonl, write_jsonl
from mtrag.experiments.artifacts import (
    CandidateStore,
    JsonlCheckpoint,
    RunArtifacts,
)
from mtrag.schemas import (
    ArtifactRef,
    BenchmarkTask,
    Message,
    SearchHit,
)


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
            with self.assertRaisesRegex(ValueError, "Invalid JSON"):
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

    def test_existing_duplicate_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "items.jsonl"
            path.write_text('{"task_id":"one"}\n{"task_id":"one"}\n')

            with self.assertRaisesRegex(ValueError, "duplicate checkpoint"):
                JsonlCheckpoint(path)


class ArtifactMappingTest(unittest.TestCase):
    revision = "a" * 64

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

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidates.jsonl"
            store = CandidateStore(path)
            store.append_hits({task.task_id: task}, {task.task_id: (hit,)})
            record = read_jsonl(path)[0]
            restored = store.rankings({task.task_id: task})[task.task_id]
            contexts = store.contexts(top_k=1)[task.task_id]

        self.assertEqual(record["contexts"][0]["score"], 1.0)
        self.assertEqual(restored.hits, (hit,))
        self.assertEqual(contexts[0].document_id, "doc")
        self.assertEqual(contexts[0].score, 12.5)

    def test_run_artifacts_separates_named_experiment_revisions(self) -> None:
        paths = RunArtifacts(Path("run"))
        retrieval = ArtifactRef("bge_last.dense", self.revision)
        generation = ArtifactRef("answer", self.revision)
        self.assertEqual(
            paths.candidates(retrieval),
            Path(
                "run/experiments/bge_last/dense/"
                f"{self.revision}/candidates.jsonl"
            ),
        )
        self.assertEqual(
            paths.generation(generation),
            Path(f"run/generation/answer/{self.revision}/predictions.jsonl"),
        )

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
            store = CandidateStore(candidates)
            store.append_hits(
                {task.task_id: task for task in tasks},
                {tasks[0].task_id: (hit,)},
            )

            count = store.write_prediction(
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

    def test_reads_existing_candidate_jsonl_format(self) -> None:
        task = BenchmarkTask(
            task_id="q<::>1",
            conversation_id="q",
            turn=1,
            collection="collection",
            domain="clapnq",
            messages=(Message("user", "question"),),
        )
        row = {
            "conversation_id": "q",
            "task_id": "q<::>1",
            "turn": 1,
            "Collection": "collection",
            "input": [{"speaker": "user", "text": "question"}],
            "contexts": [
                {
                    "document_id": "doc",
                    "score": 1.0,
                    "retriever_score": 12.5,
                    "rank": 1,
                    "source": "elser",
                    "title": "Title",
                    "text": "Passage",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidates.jsonl"
            write_jsonl(path, [row])

            loaded = CandidateStore(path).rankings({task.task_id: task})[task.task_id]

        self.assertEqual(loaded.hits[0].document_id, "doc")
        self.assertEqual(loaded.hits[0].score, 12.5)
        self.assertEqual(loaded.hits[0].source, "elser")


if __name__ == "__main__":
    unittest.main()

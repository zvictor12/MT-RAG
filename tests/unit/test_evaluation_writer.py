import json
import tempfile
import unittest
from pathlib import Path

from mtrag.evaluation.writer import (
    deterministic_rank_score,
    make_retrieval_record,
    write_retrieval_jsonl,
)


class RetrievalWriterTest(unittest.TestCase):
    def test_make_record_is_official_and_does_not_mutate_input(self) -> None:
        base = {
            "task_id": "q::1",
            "Collection": "clapnq",
            "input": [{"speaker": "user", "text": "Привет"}],
            "targets": [{"speaker": "agent", "text": "hidden"}],
        }
        contexts = [
            {"document_id": "d1", "score": 99.0, "text": "one"},
            {"document_id": "d1", "score": 98.0, "text": "duplicate"},
            {"document_id": "d2", "score": -3.0, "text": "two"},
        ]
        record = make_retrieval_record(base, contexts)

        self.assertNotIn("contexts", base)
        self.assertIn("targets", base)
        self.assertNotIn("targets", record)
        self.assertEqual(
            [context["document_id"] for context in record["contexts"]],
            ["d1", "d2"],
        )
        self.assertEqual(
            [context["score"] for context in record["contexts"]],
            [1.0, 0.5],
        )

    def test_write_jsonl_atomically(self) -> None:
        record = {
            "task_id": "q::1",
            "Collection": "clapnq",
            "contexts": [{"document_id": "d1", "score": 1.0}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "retrieval.jsonl"
            self.assertEqual(write_retrieval_jsonl(path, [record]), 1)
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                record,
            )

    def test_rank_score_rejects_non_positive_rank(self) -> None:
        self.assertEqual(deterministic_rank_score(2), 0.5)
        with self.assertRaises(ValueError):
            deterministic_rank_score(0)


if __name__ == "__main__":
    unittest.main()

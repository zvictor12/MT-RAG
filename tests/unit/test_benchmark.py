import json
import tempfile
import unittest
from pathlib import Path

from mtrag.data.benchmark import (
    BenchmarkRepository,
    clean_query_text,
    domain_for_collection,
    normalize_task_id,
)
from mtrag.schemas import QueryVariant


COLLECTION = "mt-rag-clapnq-elser-512-100-20240503"


class BenchmarkRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        human = self.root / "mtrag-human"
        generation = human / "generation_tasks"
        retrieval = human / "retrieval_tasks"
        generation.mkdir(parents=True)

        task = {
            "task_id": "conversation::2",
            "conversation_id": "conversation",
            "turn": "2",
            "Collection": COLLECTION,
            "input": [
                {"speaker": "user", "text": "What is Cloudant?"},
                {"speaker": "agent", "text": "A document database."},
                {"speaker": "user", "text": "Can it store JSON?"},
            ],
            "contexts": [
                {
                    "document_id": "doc-1",
                    "title": "Cloudant",
                    "text": "Cloudant stores JSON documents.",
                    "url": "https://example.test/cloudant",
                    "score": 3,
                }
            ],
            "targets": [{"speaker": "agent", "text": "Yes."}],
            "Answerability": ["ANSWERABLE"],
        }
        self._write_rows(generation / "reference.jsonl", [task])

        for domain in ("clapnq", "cloud", "govt", "fiqa"):
            folder = retrieval / domain
            folder.mkdir(parents=True)
            task_id = "conversation<::>2" if domain == "clapnq" else f"{domain}<::>1"
            self._write_rows(
                folder / f"{domain}_lastturn.jsonl",
                [{"_id": task_id, "text": "|user|: last question"}],
            )
            self._write_rows(
                folder / f"{domain}_rewrite.jsonl",
                [{"_id": task_id, "text": "|user|: standalone question"}],
            )

        self.repository = BenchmarkRepository(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_loads_generation_task_and_collection_domain(self) -> None:
        task = self.repository.load_tasks()[0]

        self.assertEqual(task.task_id, "conversation<::>2")
        self.assertEqual(task.turn, 2)
        self.assertEqual(task.collection, COLLECTION)
        self.assertEqual(task.domain, "clapnq")
        self.assertEqual(task.contexts[0].url, "https://example.test/cloudant")
        self.assertEqual(task.targets[0].text, "Yes.")

    def test_loads_official_last_and_gold_variants(self) -> None:
        last = self.repository.query_cases(QueryVariant.LAST)
        gold = self.repository.query_cases("gold")

        self.assertEqual(len(last), 4)
        self.assertEqual(last[0].text, "last question")
        self.assertEqual(gold[0].text, "standalone question")

    def test_qwen_queries_are_external_and_legacy_ids_are_normalized(self) -> None:
        cases = self.repository.query_cases(
            QueryVariant.QWEN,
            qwen_queries={"conversation::2": "Qwen standalone question"},
        )

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].task_id, "conversation<::>2")
        self.assertEqual(cases[0].domain, "clapnq")

    def test_qwen_queries_can_be_read_from_jsonl(self) -> None:
        path = self.root / "qwen.jsonl"
        self._write_rows(
            path,
            [{"task_id": "conversation::2", "query": "from file"}],
        )

        cases = self.repository.query_cases("qwen", qwen_queries=path)

        self.assertEqual(cases[0].text, "from file")

    @staticmethod
    def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")


class BenchmarkHelpersTests(unittest.TestCase):
    def test_normalizes_only_a_numeric_legacy_turn_suffix(self) -> None:
        self.assertEqual(normalize_task_id("abc::12"), "abc<::>12")
        self.assertEqual(normalize_task_id("abc<::>12"), "abc<::>12")
        self.assertEqual(normalize_task_id("namespace::label"), "namespace::label")

    def test_collection_mapping_and_query_cleanup(self) -> None:
        self.assertEqual(domain_for_collection(COLLECTION), "clapnq")
        self.assertEqual(clean_query_text(" |user|:  question "), "question")


if __name__ == "__main__":
    unittest.main()

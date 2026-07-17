import tempfile
import unittest
from pathlib import Path

from mtrag.evaluation.qrels import (
    load_qrels,
    normalize_domain,
    normalize_task_id,
    qrels_path,
)


class QrelsTest(unittest.TestCase):
    def test_normalize_only_trailing_legacy_delimiter(self) -> None:
        self.assertEqual(normalize_task_id("abc::2"), "abc<::>2")
        self.assertEqual(normalize_task_id("abc::part::2"), "abc::part<::>2")
        self.assertEqual(normalize_task_id("abc<::>2"), "abc<::>2")
        self.assertEqual(normalize_task_id("https://example.test/a::b"), "https://example.test/a::b")

    def test_domain_aliases_and_canonical_path(self) -> None:
        self.assertEqual(normalize_domain("ibmcloud"), "cloud")
        self.assertEqual(
            normalize_domain("mt-rag-clapnq-bge-m3-dense"),
            "clapnq",
        )
        self.assertEqual(
            qrels_path("/benchmark", "cloud"),
            Path("/benchmark/mtrag-human/retrieval_tasks/cloud/qrels/dev.tsv"),
        )

    def test_load_qrels_normalizes_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dev.tsv"
            path.write_text(
                "query-id\tcorpus-id\tscore\n"
                "q::1\td1\t1\n"
                "q<::>1\td2\t1\n",
                encoding="utf-8",
            )
            self.assertEqual(
                load_qrels(path),
                {"q<::>1": {"d1": 1, "d2": 1}},
            )

    def test_conflicting_duplicate_qrel_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dev.tsv"
            path.write_text(
                "query-id\tcorpus-id\tscore\n"
                "q::1\td1\t1\n"
                "q<::>1\td1\t0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "conflicting qrel"):
                load_qrels(path)


if __name__ == "__main__":
    unittest.main()

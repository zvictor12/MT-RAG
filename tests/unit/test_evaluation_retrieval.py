import json
import math
import tempfile
import unittest
from pathlib import Path

from mtrag.evaluation.retrieval import (
    evaluate_retrieval,
    load_rankings_jsonl,
    rank_document_ids,
    score_query,
)


class RetrievalEvaluationTest(unittest.TestCase):
    def test_query_metrics(self) -> None:
        metrics = score_query(
            {"d1": 1, "d2": 1},
            ["noise", "d1"],
            cutoffs=(1, 3),
        )
        expected_ndcg = (1 / math.log2(3)) / (
            1 + 1 / math.log2(3)
        )
        self.assertEqual(metrics.ndcg[1], 0.0)
        self.assertAlmostEqual(metrics.ndcg[3], expected_ndcg)
        self.assertEqual(metrics.recall[1], 0.0)
        self.assertEqual(metrics.recall[3], 0.5)

    def test_missing_queries_are_zero_and_global_is_qrels_weighted(self) -> None:
        qrels = {
            "clapnq": {
                "q1<::>1": {"d1": 1},
                "q2<::>1": {"d2": 1},
            },
            "cloud": {"q3<::>1": {"d3": 1}},
        }
        rankings = {
            "clapnq": {"q1::1": ["d1"]},
            "cloud": {"q3::1": ["d3"]},
        }
        report = evaluate_retrieval(qrels, rankings, cutoffs=(1,))

        self.assertEqual(report.query_count, 3)
        self.assertEqual(report.domains["clapnq"].query_count, 2)
        self.assertEqual(report.domains["clapnq"].metrics.ndcg[1], 0.5)
        self.assertEqual(report.metrics.ndcg[1], 2 / 3)
        self.assertEqual(report.metrics.recall[1], 2 / 3)

    def test_score_ties_have_deterministic_document_id_order(self) -> None:
        self.assertEqual(
            rank_document_ids({"b": 1.0, "a": 1.0}),
            ["a", "b"],
        )
        report = evaluate_retrieval(
            {"fiqa": {"q<::>1": {"a": 1}}},
            {"fiqa": {"q::1": {"b": 1.0, "a": 1.0}}},
            cutoffs=(1,),
        )
        self.assertEqual(report.metrics.ndcg[1], 1.0)

    def test_ordered_rankings_are_deduplicated(self) -> None:
        self.assertEqual(rank_document_ids(["a", "a", "b"]), ["a", "b"])

    def test_load_official_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "task_id": "q::1",
                        "Collection": "mt-rag-ibmcloud-elser-512-100-20240502",
                        "contexts": [
                            {"document_id": "d1", "score": 2.0},
                            {"document_id": "d2", "score": 1.0},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                load_rankings_jsonl(path),
                {"cloud": {"q<::>1": {"d1": 2.0, "d2": 1.0}}},
            )


if __name__ == "__main__":
    unittest.main()

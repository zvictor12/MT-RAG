import unittest

from mtrag.evaluation.generation import (
    AlgorithmicGenerationEvaluator,
    _clean_for_extractiveness,
    summarize_generation_metrics,
)


class FakeSemanticScorer:
    def __init__(self) -> None:
        self.calls = 0
        self.pair_count = 0

    def score(self, candidates, references):
        self.calls += 1
        self.pair_count = len(candidates)
        values = [0.5] * len(references)
        return values, values, values


class GenerationEvaluationTest(unittest.TestCase):
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

    def test_batches_all_target_and_passage_pairs_once(self) -> None:
        scorer = FakeSemanticScorer()
        evaluator = AlgorithmicGenerationEvaluator(
            scorer,
            rouge_l=lambda _prediction, _target: 0.25,
        )
        records = [
            {
                "task_id": "q",
                "predictions": [{"text": "the answer"}],
                "targets": [{"text": "answer"}],
                "contexts": [{"text": "context one"}, {"text": "context two"}],
            }
        ]

        output = evaluator.evaluate(records)

        self.assertEqual(scorer.calls, 1)
        self.assertEqual(scorer.pair_count, 3)
        self.assertEqual(output[0]["metrics"]["BertscoreR"], [0.5])
        self.assertEqual(output[0]["metrics"]["BertKPrec"], [0.5, 0.5])
        self.assertGreater(output[0]["metrics"]["RB_agg"][0], 0)

    def test_extractiveness_cleanup_matches_official_operation_order(self) -> None:
        self.assertEqual(
            _clean_for_extractiveness("<p>Hello, world!</p>"),
            "phello worldp",
        )


if __name__ == "__main__":
    unittest.main()

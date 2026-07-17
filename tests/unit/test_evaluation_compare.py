import unittest

from mtrag.evaluation.compare import paired_bootstrap
from mtrag.evaluation.retrieval import evaluate_retrieval


class PairedBootstrapTest(unittest.TestCase):
    def test_is_deterministic_and_paired(self) -> None:
        qrels = {
            "clapnq": {
                "q1<::>1": {"d1": 1},
                "q2<::>1": {"d2": 1},
                "q3<::>1": {"d3": 1},
            }
        }
        baseline = evaluate_retrieval(
            qrels,
            {"clapnq": {"q1::1": ["d1"]}},
            cutoffs=(1,),
        )
        candidate = evaluate_retrieval(
            qrels,
            {
                "clapnq": {
                    "q1::1": ["d1"],
                    "q2::1": ["d2"],
                    "q3::1": ["d3"],
                }
            },
            cutoffs=(1,),
        )

        first = paired_bootstrap(
            baseline,
            candidate,
            cutoff=1,
            samples=500,
            seed=7,
        )
        second = paired_bootstrap(
            baseline,
            candidate,
            cutoff=1,
            samples=500,
            seed=7,
        )

        self.assertEqual(first, second)
        self.assertAlmostEqual(first.difference, 2 / 3)
        self.assertEqual(first.query_count, 3)
        self.assertGreater(first.probability_improvement, 0.9)


if __name__ == "__main__":
    unittest.main()

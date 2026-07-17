import json
import tempfile
import unittest
from pathlib import Path

from mtrag.experiments.reporting import render_experiment_results


class ExperimentReportingTests(unittest.TestCase):
    def test_renders_retrieval_generation_and_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            retrieval = run_dir / "evaluation" / "retrieval"
            generation = run_dir / "evaluation" / "generation"
            decisions = run_dir / "decisions"
            retrieval.mkdir(parents=True)
            generation.mkdir(parents=True)
            decisions.mkdir()

            (retrieval / "dense.json").write_text(
                json.dumps(
                    {
                        "query_count": 2,
                        "metrics": {
                            "ndcg": {str(k): 0.1 for k in (1, 3, 5, 10)},
                            "recall": {str(k): 0.2 for k in (1, 3, 5, 10)},
                        },
                    }
                ),
                encoding="utf-8",
            )
            (generation / "task_b.json").write_text(
                json.dumps(
                    {
                        "task_count": 2,
                        "metrics": {
                            name: {"mean": 0.3}
                            for name in (
                                "Recall",
                                "RougeL_stemFalse",
                                "BertscoreP",
                                "BertscoreR",
                                "BertKPrec",
                                "Extractiveness_RougeL",
                                "RB_agg",
                                "Length",
                            )
                        },
                    }
                ),
                encoding="utf-8",
            )
            (decisions / "bge-winner.json").write_text(
                json.dumps(
                    {
                        "winner": "dense",
                        "metric": "ndcg@5",
                        "score": 0.1,
                    }
                ),
                encoding="utf-8",
            )
            (decisions / "rewrite-winner.json").write_text(
                json.dumps(
                    {
                        "winner": "qwen_t02",
                        "metric": "ndcg@5",
                        "score": 0.2,
                    }
                ),
                encoding="utf-8",
            )
            (decisions / "reranker-variants.json").write_text(
                json.dumps(
                    {
                        "enabled": True,
                        "variants": {
                            "qwen_t02": {
                                "enabled": True,
                                "ndcg5_gain": 0.03,
                                "probability_improvement": 0.97,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            output = render_experiment_results(run_dir)

        self.assertIn("RETRIEVAL (queries: 2)", output)
        self.assertIn("dense", output)
        self.assertIn("GENERATION", output)
        self.assertIn("task_b", output)
        self.assertIn("BGE winner: dense (ndcg@5=0.1000)", output)
        self.assertIn("Rewrite winner: qwen_t02 (ndcg@5=0.2000)", output)
        self.assertIn("Reranker qwen_t02: enabled", output)


if __name__ == "__main__":
    unittest.main()

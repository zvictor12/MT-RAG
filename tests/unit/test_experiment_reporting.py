import json
import tempfile
import unittest
from pathlib import Path

from mtrag.experiments.reporting import render_experiment_results


class ExperimentReportingTests(unittest.TestCase):
    def test_renders_every_revision_without_a_winner_section(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            retrieval = (
                run_dir
                / "experiments/bge_last/dense"
                / ("a" * 64)
                / "evaluation"
                / ("b" * 64)
            )
            generation = (
                run_dir
                / "generation/task_c"
                / ("c" * 64)
                / "evaluation"
                / ("d" * 64)
            )
            retrieval.mkdir(parents=True)
            generation.mkdir(parents=True)
            (retrieval / "task-a-metrics.json").write_text(
                json.dumps(
                    {
                        "query_count": 2,
                        "metrics": {
                            "ndcg": {str(k): 0.1 for k in (1, 3, 5, 10)},
                            "recall": {str(k): 0.2 for k in (1, 3, 5, 10)},
                        },
                    }
                )
            )
            (generation / "ibm-summary.json").write_text(
                json.dumps(
                    {
                        "task_count": 2,
                        "metrics": {"RB_agg": {"mean": 0.3}},
                    }
                )
            )

            output = render_experiment_results(run_dir)

        self.assertIn("bge_last.dense@aaaaaaaa/eval@bbbbbbbb", output)
        self.assertIn("task_c@cccccccc/eval@dddddddd", output)
        self.assertIn("0.3000", output)
        self.assertNotIn("winner", output.lower())
        self.assertNotIn("DECISIONS", output)


if __name__ == "__main__":
    unittest.main()

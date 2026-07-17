import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from mtrag.evaluation.ibm import load_ibm_module
from mtrag.evaluation.retrieval import evaluate_retrieval


CLAPNQ = "mt-rag-clapnq-elser-512-100-20240503"
CLOUD = "mt-rag-ibmcloud-elser-512-100-20240502"


def metric_scores(query_id: str, value: float, cutoffs: list[int]) -> dict:
    return {
        query_id: {
            **{f"ndcg_cut_{cutoff}": value for cutoff in cutoffs},
            **{f"recall_{cutoff}": value / 2 for cutoff in cutoffs},
        }
    }


class FakeOfficialEvaluator:
    def __init__(self) -> None:
        self.qrels_paths: list[Path] = []
        self.evaluated: list[tuple[dict, list[int]]] = []

    def prepare_results_dict(self, path: str):
        self.prediction_path = Path(path)
        return (
            {
                "clap-q": {"d1": 1.0},
                "clap-extra-1": {"d2": 1.0},
                "clap-extra-2": {"d3": 1.0},
                "cloud-q": {"d4": 1.0},
            },
            {
                "clap-q": CLAPNQ,
                "clap-extra-1": CLAPNQ,
                "clap-extra-2": CLAPNQ,
                "cloud-q": CLOUD,
            },
        )

    def load_qrels(self, path: str):
        qrels_path = Path(path)
        self.qrels_paths.append(qrels_path)
        return {"domain": qrels_path.parts[-3]}

    def evaluate(self, qrels: dict, results: dict, cutoffs: list[int]):
        self.evaluated.append((results, cutoffs))
        domain = qrels["domain"]
        value = 0.2 if domain == "clapnq" else 0.8
        query_id = "clap-q" if domain == "clapnq" else "cloud-q"
        return (
            metric_scores(query_id, value, cutoffs),
            {f"NDCG@{cutoff}": value for cutoff in cutoffs},
            {},
            {f"Recall@{cutoff}": value / 2 for cutoff in cutoffs},
            {},
        )


class RetrievalEvaluationTest(unittest.TestCase):
    def test_uses_ibm_functions_and_official_prediction_weighting(self) -> None:
        official = FakeOfficialEvaluator()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "benchmark"
            prediction = Path(directory) / "task-a.jsonl"
            with patch(
                "mtrag.evaluation.retrieval.load_ibm_module",
                return_value=official,
            ):
                report = evaluate_retrieval(root, prediction)

        self.assertEqual(official.prediction_path, prediction)
        self.assertEqual(
            official.qrels_paths,
            [
                root
                / "mtrag-human/retrieval_tasks/clapnq/qrels/dev.tsv",
                root
                / "mtrag-human/retrieval_tasks/cloud/qrels/dev.tsv",
            ],
        )
        self.assertEqual(len(official.evaluated), 2)
        self.assertTrue(
            all(cutoffs == [1, 3, 5, 10] for _, cutoffs in official.evaluated)
        )
        self.assertEqual(report.query_count, 2)
        self.assertEqual(report.domains["clapnq"].query_count, 1)
        self.assertEqual(report.domains["clapnq"].metrics.ndcg[5], 0.2)
        self.assertEqual(report.domains["cloud"].query_count, 1)

        # IBM weights domain means by prediction count: (0.2 * 3 + 0.8) / 4.
        self.assertAlmostEqual(report.metrics.ndcg[5], 0.35)
        self.assertAlmostEqual(report.metrics.recall[5], 0.175)

    def test_loader_executes_a_script_from_the_benchmark_repository(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / "scripts" / "evaluation"
            scripts.mkdir(parents=True)
            (scripts / "sentinel.py").write_text("VALUE = 42\n", encoding="utf-8")

            module = load_ibm_module(root, "sentinel.py")

        self.assertEqual(module.VALUE, 42)

    def test_loader_temporarily_overrides_an_upstream_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / "scripts" / "evaluation"
            scripts.mkdir(parents=True)
            (scripts / "sentinel.py").write_text(
                "import external_metric\nVALUE = external_metric.VALUE\n",
                encoding="utf-8",
            )
            dependency = ModuleType("external_metric")
            dependency.VALUE = 7

            module = load_ibm_module(
                root,
                "sentinel.py",
                module_overrides={"external_metric": dependency},
            )

        self.assertEqual(module.VALUE, 7)
        self.assertNotIn("external_metric", sys.modules)


if __name__ == "__main__":
    unittest.main()

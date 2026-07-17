import tempfile
import unittest
from pathlib import Path

from mtrag.experiments.spec import ExperimentConfig


MINIMAL_CONFIG = """
[run]
name = "pilot"
output_root = "runs"
benchmark_root = "../benchmark"
cpu_slots = 4

[models]
bge_path = "models/bge"
reranker_path = "models/reranker"

[retrieval]
rrf_top_k = 20
prediction_top_k = 10

[reranking]
input_top_k = 20
output_top_k = 10
"""


class ExperimentConfigTest(unittest.TestCase):
    def test_relative_paths_are_resolved_from_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "configs" / "pilot.toml"
            path.parent.mkdir()
            path.write_text(MINIMAL_CONFIG)

            config = ExperimentConfig.load(path)

            self.assertEqual(config.project_root, root)
            self.assertEqual(config.default_run_dir, root / "runs" / "pilot")
            self.assertEqual(config.models.bge_path, root / "models" / "bge")
            self.assertEqual(config.run.benchmark_root, root / "../benchmark")

    def test_official_prediction_limit_is_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "configs" / "pilot.toml"
            path.parent.mkdir()
            path.write_text(MINIMAL_CONFIG.replace("prediction_top_k = 10", "prediction_top_k = 11"))

            with self.assertRaisesRegex(ValueError, "official limit"):
                ExperimentConfig.load(path)

    def test_rewrite_variant_temperatures_are_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "configs" / "pilot.toml"
            path.parent.mkdir()
            path.write_text(
                MINIMAL_CONFIG
                + """
[rewriting.variants.qwen_t0]
temperature = 0.0

[rewriting.variants.qwen_t02]
temperature = 0.2
"""
            )

            config = ExperimentConfig.load(path)

            self.assertEqual(config.rewriting.variant("qwen_t0").temperature, 0.0)
            self.assertEqual(config.rewriting.variant("qwen_t02").temperature, 0.2)


if __name__ == "__main__":
    unittest.main()

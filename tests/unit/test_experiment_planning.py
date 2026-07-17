import tempfile
import unittest
from pathlib import Path

from mtrag.experiments.planning import build_plan
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


class ExperimentPlanTest(unittest.TestCase):
    def test_plan_serializes_gpu_work_and_preserves_artifact_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "configs" / "pilot.toml"
            config_path.parent.mkdir()
            config_path.write_text(MINIMAL_CONFIG)
            (root / "scripts").mkdir()
            config = ExperimentConfig.load(config_path)

            plan = build_plan(config, root / "run", phase="full")
            stages = {stage.name: stage for stage in plan}

            self.assertEqual(stages["encode_bge"].dependencies, ("rewrite_qwen",))
            self.assertEqual(
                stages["encode_bge_variants"].dependencies,
                ("encode_bge", "retrieve_bge", "rewrite_qwen_t02"),
            )
            self.assertEqual(
                stages["retrieve_elser"].dependencies,
                ("select_rewrite_variant",),
            )
            self.assertTrue(stages["encode_bge"].resources.gpu)
            self.assertFalse(stages["retrieve_elser"].resources.gpu)
            self.assertEqual(stages["retrieve_elser"].resources.cpu_slots, 3)
            self.assertEqual(
                stages["decide_reranker"].dependencies,
                ("evaluate_bge_base", "evaluate_bge_rerank"),
            )
            self.assertEqual(
                stages["rerank_elser"].dependencies,
                ("decide_bge_variants", "retrieve_elser"),
            )
            self.assertIsNotNone(stages["rerank_elser"].condition)
            self.assertEqual(
                stages["generate_task_b"].dependencies,
                ("select_winner",),
            )
            self.assertEqual(
                stages["generate_task_c_bge"].dependencies,
                ("generate_task_b", "select_bge"),
            )
            self.assertIsNotNone(stages["evaluate_generation_bge"].condition)
            self.assertEqual(
                stages["select_winner"].dependencies,
                (
                    "select_bge_variants",
                    "evaluate_elser_base",
                    "evaluate_elser_rerank",
                ),
            )
            self.assertEqual(
                stages["generate_task_c"].dependencies,
                ("generate_task_b", "select_winner"),
            )
            self.assertIsNotNone(stages["evaluate_generation"].condition)

    def test_bge_is_default_and_full_is_a_strict_superset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "configs" / "pilot.toml"
            config_path.parent.mkdir()
            config_path.write_text(MINIMAL_CONFIG)
            config = ExperimentConfig.load(config_path)
            run_dir = root / "run"

            bge = build_plan(config, run_dir)
            full = build_plan(config, run_dir, phase="full")
            bge_names = tuple(stage.name for stage in bge)
            full_names = tuple(stage.name for stage in full)

            self.assertLess(set(bge_names), set(full_names))
            self.assertEqual(full_names[: len(bge_names)], bge_names)
            self.assertIn("select_bge", bge_names)
            self.assertIn("select_rewrite_variant", bge_names)
            self.assertIn("select_bge_variants", bge_names)
            self.assertIn("generate_task_c_bge", bge_names)
            self.assertIn("generate_task_c_bge_last", bge_names)
            self.assertIn("generate_task_c_bge_selected", bge_names)
            self.assertIn("evaluate_generation_bge", bge_names)
            self.assertNotIn("retrieve_elser", bge_names)
            self.assertNotIn("select_winner", bge_names)
            self.assertNotIn("generate_task_c", bge_names)

    def test_rejects_unknown_phase(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "configs" / "pilot.toml"
            config_path.parent.mkdir()
            config_path.write_text(MINIMAL_CONFIG)
            config = ExperimentConfig.load(config_path)

            with self.assertRaisesRegex(ValueError, "unknown experiment phase"):
                build_plan(config, root / "run", phase="elser")

    def test_every_stage_invokes_the_same_runner_and_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "configs" / "pilot.toml"
            config_path.parent.mkdir()
            config_path.write_text(MINIMAL_CONFIG)
            config = ExperimentConfig.load(config_path)
            run_dir = root / "run"

            for stage in build_plan(config, run_dir):
                self.assertIn(str(root / "scripts" / "run_experiment.py"), stage.command)
                self.assertIn(stage.name, stage.command)
                self.assertIn(str(run_dir.resolve()), stage.command)


if __name__ == "__main__":
    unittest.main()

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from mtrag.experiments.artifacts import RunArtifacts
from mtrag.experiments.spec import ExperimentConfig
from mtrag.llm.prompts import GENERATOR_PROMPT_VERSION, REWRITE_PROMPT_VERSION
from scripts.run_experiment import CONFIG_DIGEST_SCOPE, lock_definition


CONFIG = """
[run]
name = "main"
output_root = "runs"
benchmark_root = "../benchmark"
cpu_slots = 4

[thermal]
gpu_pause = {gpu_pause}
gpu_resume = 72.0
gpu_abort = 86.0
cpu_pause = 90.0
cpu_resume = 80.0
cpu_abort = 97.0
"""

REWRITE_VARIANTS = """

[rewriting.variants.qwen_t0]
temperature = 0.0

[rewriting.variants.qwen_t02]
temperature = {temperature}
"""


class ExperimentDefinitionTest(unittest.TestCase):
    def test_thermal_changes_do_not_invalidate_a_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "configs" / "experiment.toml"
            config_path.parent.mkdir()
            config_path.write_text(CONFIG.format(gpu_pause=80.0))
            run_dir = root / "run"

            lock_definition(ExperimentConfig.load(config_path), run_dir)
            before = (run_dir / "run-definition.json").read_text()
            config_path.write_text(CONFIG.format(gpu_pause=86.0))
            lock_definition(ExperimentConfig.load(config_path), run_dir)

            self.assertEqual((run_dir / "run-definition.json").read_text(), before)

    def test_output_affecting_changes_still_invalidate_a_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "configs" / "experiment.toml"
            config_path.parent.mkdir()
            config_path.write_text(CONFIG.format(gpu_pause=80.0))
            run_dir = root / "run"
            lock_definition(ExperimentConfig.load(config_path), run_dir)

            changed = CONFIG.replace("cpu_slots = 4", "cpu_slots = 3")
            config_path.write_text(changed.format(gpu_pause=80.0))

            with self.assertRaisesRegex(RuntimeError, "new --run-dir"):
                lock_definition(ExperimentConfig.load(config_path), run_dir)

    def test_legacy_definition_is_migrated_when_only_thermal_changed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "configs" / "experiment.toml"
            config_path.parent.mkdir()
            old_config = CONFIG.format(gpu_pause=80.0)
            config_path.write_text(CONFIG.format(gpu_pause=86.0))
            artifacts = RunArtifacts(root / "run")
            artifacts.root.mkdir()
            artifacts.config_snapshot.write_text(old_config)
            artifacts.definition.write_text(
                json.dumps(
                    {
                        "config_sha256": hashlib.sha256(
                            old_config.encode()
                        ).hexdigest(),
                        "rewrite_prompt_version": REWRITE_PROMPT_VERSION,
                        "generator_prompt_version": GENERATOR_PROMPT_VERSION,
                    }
                )
            )

            lock_definition(ExperimentConfig.load(config_path), artifacts.root)

            definition = json.loads(artifacts.definition.read_text())
            self.assertEqual(
                definition["config_digest_scope"],
                CONFIG_DIGEST_SCOPE,
            )

    def test_rewrite_temperature_is_part_of_the_definition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "configs" / "experiment.toml"
            config_path.parent.mkdir()
            base = CONFIG.format(gpu_pause=80.0)
            config_path.write_text(
                base + REWRITE_VARIANTS.format(temperature=0.2)
            )
            run_dir = root / "run"
            lock_definition(ExperimentConfig.load(config_path), run_dir)

            config_path.write_text(
                base + REWRITE_VARIANTS.format(temperature=0.3)
            )

            with self.assertRaisesRegex(RuntimeError, "new --run-dir"):
                lock_definition(ExperimentConfig.load(config_path), run_dir)

    def test_existing_run_accepts_only_the_added_rewrite_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "configs" / "experiment.toml"
            config_path.parent.mkdir()
            old_config = CONFIG.format(gpu_pause=80.0)
            config_path.write_text(old_config)
            run_dir = root / "run"
            lock_definition(ExperimentConfig.load(config_path), run_dir)

            extended = old_config + REWRITE_VARIANTS.format(temperature=0.2)
            config_path.write_text(extended)
            lock_definition(ExperimentConfig.load(config_path), run_dir)
            lock_definition(ExperimentConfig.load(config_path), run_dir)

            artifacts = RunArtifacts(run_dir)
            self.assertEqual(artifacts.config_snapshot.read_text(), extended)

    def test_variant_migration_rejects_an_unrelated_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "configs" / "experiment.toml"
            config_path.parent.mkdir()
            old_config = CONFIG.format(gpu_pause=80.0)
            config_path.write_text(old_config)
            run_dir = root / "run"
            lock_definition(ExperimentConfig.load(config_path), run_dir)

            changed = old_config.replace("cpu_slots = 4", "cpu_slots = 3")
            config_path.write_text(
                changed + REWRITE_VARIANTS.format(temperature=0.2)
            )

            with self.assertRaisesRegex(RuntimeError, "new --run-dir"):
                lock_definition(ExperimentConfig.load(config_path), run_dir)


if __name__ == "__main__":
    unittest.main()

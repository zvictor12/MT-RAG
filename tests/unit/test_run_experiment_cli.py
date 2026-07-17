import tempfile
import unittest
from pathlib import Path

from scripts.run_experiment import _campaign_lock, _config_snapshot


class RunExperimentCliTest(unittest.TestCase):
    def test_config_snapshot_is_stable_and_keeps_the_project_base(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "configs" / "experiment.toml"
            source.parent.mkdir()
            source.write_text("value = 1\n", encoding="utf-8")

            with _config_snapshot(source) as snapshot:
                self.assertEqual(snapshot.parent, source.parent)
                source.write_text("value = 2\n", encoding="utf-8")
                self.assertEqual(snapshot.read_text(encoding="utf-8"), "value = 1\n")

            self.assertFalse(snapshot.exists())

    def test_campaign_allows_only_one_scheduler(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            with _campaign_lock(run_dir):
                with self.assertRaisesRegex(RuntimeError, "another scheduler"):
                    with _campaign_lock(run_dir):
                        self.fail("a second scheduler acquired the same campaign")


if __name__ == "__main__":
    unittest.main()

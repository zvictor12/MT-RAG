import json
import tempfile
import unittest
from pathlib import Path

from mtrag.experiments.artifacts import RunArtifacts
from mtrag.experiments.planning import PlannedStage
from mtrag.experiments.stages import _is_complete


class ExperimentStageCompletionTest(unittest.TestCase):
    def test_corrupt_marker_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            revision = "a" * 64
            stage = PlannedStage(
                name=f"retrieve.example.dense.{revision[:12]}",
                kind="retrieve",
                fingerprint=revision,
                params={"reference": "example.dense", "revision": revision},
            )
            artifacts = RunArtifacts(Path(directory))
            artifacts.create_directories()
            output = artifacts.candidates("example.dense", revision)
            output.parent.mkdir(parents=True)
            output.touch()
            marker = artifacts.stage_marker(revision)
            marker.write_text("not json", encoding="utf-8")

            self.assertFalse(_is_complete(stage, artifacts))

            marker.write_text(
                json.dumps(
                    {
                        "stage": stage.name,
                        "kind": stage.kind,
                        "fingerprint": revision,
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(_is_complete(stage, artifacts))

if __name__ == "__main__":
    unittest.main()

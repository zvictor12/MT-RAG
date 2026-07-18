import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from mtrag.experiments.artifacts import RunArtifacts
from mtrag.experiments.planning import PlannedStage, Workflow
from mtrag.experiments.stages import run_stage


class ExperimentStageCompletionTest(unittest.TestCase):
    def test_completed_marker_skips_the_handler(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            revision = "a" * 64
            stage = PlannedStage(
                name=f"retrieve.example.dense.{revision[:12]}",
                kind="retrieve",
                fingerprint=revision,
                params={},
            )
            artifacts = RunArtifacts(Path(directory))
            marker = artifacts.stage_marker(revision)
            marker.parent.mkdir(parents=True)
            marker.write_text("not json", encoding="utf-8")
            handler = Mock()

            with patch.dict(
                "mtrag.experiments.stages.EXECUTORS",
                {"retrieve": handler},
            ):
                run_stage(
                    stage.name,
                    SimpleNamespace(),
                    artifacts,
                    workflow=Workflow((stage,)),
                )

            handler.assert_not_called()


if __name__ == "__main__":
    unittest.main()

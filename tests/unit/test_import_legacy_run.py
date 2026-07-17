import json
import tempfile
import unittest
from pathlib import Path

from mtrag.experiments.artifacts import RunArtifacts
from mtrag.experiments.planning import PlannedStage, Workflow
from scripts.import_legacy_run import _copy_jsonl, _import_generations, _mark


class LegacyImportTest(unittest.TestCase):
    revision = "a" * 64

    def test_jsonl_is_copied_after_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "legacy.jsonl"
            target = root / "current" / "queries.jsonl"
            source.write_text(
                json.dumps({"task_id": "one", "query": "question"}) + "\n"
            )

            _copy_jsonl(source, target, {"one"}, ("query",))

            self.assertEqual(target.read_bytes(), source.read_bytes())
            self.assertNotEqual(target.stat().st_ino, source.stat().st_ino)

    def test_incompatible_jsonl_is_not_imported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "legacy.jsonl"
            target = root / "current.jsonl"
            source.write_text(
                json.dumps({"task_id": "one", "query": "question"}) + "\n"
            )

            with self.assertRaisesRegex(RuntimeError, "incomplete or incompatible"):
                _copy_jsonl(source, target, {"one", "two"}, ("query",))

            self.assertFalse(target.exists())

    def test_different_existing_target_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "legacy.jsonl"
            target = root / "current.jsonl"
            source.write_text(
                json.dumps({"task_id": "one", "query": "new"}) + "\n"
            )
            target.write_text(
                json.dumps({"task_id": "one", "query": "old"}) + "\n"
            )

            with self.assertRaisesRegex(RuntimeError, "already differs"):
                _copy_jsonl(source, target, {"one"}, ("query",))

            self.assertIn('"old"', target.read_text())

    def test_completion_marker_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts = RunArtifacts(Path(directory))
            stage = PlannedStage(
                name="rewrite.test@aaaaaaaa",
                kind="rewrite",
                fingerprint=self.revision,
                params={},
            )

            _mark(artifacts, stage)

            marker = json.loads(artifacts.stage_marker(self.revision).read_text())
            self.assertEqual(marker["stage"], stage.name)
            self.assertTrue(marker["imported_from_legacy"])

    def test_legacy_task_c_is_imported_as_an_evaluation_job(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts = RunArtifacts(Path(directory))
            source = artifacts.root / "predictions" / "task_c_bge.jsonl"
            source.parent.mkdir(parents=True)
            source.write_text(
                json.dumps({"task_id": "one", "predictions": [{"text": "answer"}]})
                + "\n"
            )
            stage = PlannedStage(
                name="generate.task_c_bge_t0_legacy@aaaaaaaa",
                kind="generate",
                fingerprint=self.revision,
                params={
                    "job_name": "task_c_bge_t0_legacy",
                    "revision": self.revision,
                },
            )

            _import_generations(artifacts, Workflow((stage,)), {"one"})

            target = artifacts.generation("task_c_bge_t0_legacy", self.revision)
            self.assertEqual(target.read_bytes(), source.read_bytes())
            self.assertTrue(artifacts.stage_marker(self.revision).is_file())


if __name__ == "__main__":
    unittest.main()

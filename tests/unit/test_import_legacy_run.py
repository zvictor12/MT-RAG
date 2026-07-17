import json
import tempfile
import unittest
from pathlib import Path

from mtrag.experiments.artifacts import RunArtifacts
from mtrag.experiments.planning import PlannedStage
from scripts.import_legacy_run import _copy_jsonl, _mark


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


if __name__ == "__main__":
    unittest.main()

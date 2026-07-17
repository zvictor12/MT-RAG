import json
import tempfile
import unittest
from pathlib import Path

from mtrag.experiments.artifacts import RunArtifacts
from mtrag.experiments.planning import PlannedStage
from scripts.import_legacy_run import _copy_jsonl, _mark


class LegacyImportTest(unittest.TestCase):
    revision = "a" * 64

    def test_jsonl_is_copied_and_validated_without_a_hardlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "legacy.jsonl"
            target = root / "current" / "queries.jsonl"
            source.write_text(
                json.dumps({"task_id": "one", "query": "question"}) + "\n"
            )

            _copy_jsonl(
                source,
                target,
                expected_ids={"one"},
                required_fields=("query",),
            )

            self.assertEqual(target.read_bytes(), source.read_bytes())
            self.assertNotEqual(target.stat().st_ino, source.stat().st_ino)
            self.assertEqual(target.stat().st_mode & 0o222, 0)

    def test_incomplete_jsonl_is_not_imported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "legacy.jsonl"
            target = root / "current.jsonl"
            source.write_text(
                json.dumps({"task_id": "one", "query": "question"}) + "\n"
            )

            with self.assertRaisesRegex(RuntimeError, "incomplete or incompatible"):
                _copy_jsonl(
                    source,
                    target,
                    expected_ids={"one", "two"},
                    required_fields=("query",),
                )

            self.assertFalse(target.exists())

    def test_completion_marker_requires_nonempty_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "legacy.jsonl"
            target = root / "empty.jsonl"
            source.write_text("source")
            target.touch()
            artifacts = RunArtifacts(root)
            stage = PlannedStage(
                name="rewrite.test@aaaaaaaa",
                kind="rewrite",
                fingerprint=self.revision,
                params={},
            )

            with self.assertRaisesRegex(RuntimeError, "incomplete target"):
                _mark(
                    artifacts,
                    stage,
                    sources=(source,),
                    targets=(target,),
                )

            self.assertFalse(artifacts.stage_marker(self.revision).exists())


if __name__ == "__main__":
    unittest.main()

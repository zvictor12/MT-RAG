import json
import tempfile
import unittest
from pathlib import Path

from mtrag.runtime.state import StageStatus, StateStore


class StateStoreTest(unittest.TestCase):
    def test_resume_reconsiders_every_active_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            first = StateStore(run_dir, ("done", "active"), resume=False)
            first.transition("done", StageStatus.RUNNING, pid=10)
            first.transition("done", StageStatus.SUCCEEDED, return_code=0)
            first.transition("active", StageStatus.RUNNING, pid=11)

            resumed = StateStore(run_dir, ("done", "active"), resume=True)

            self.assertEqual(
                resumed.manifest.stages["done"].status,
                StageStatus.PENDING,
            )
            self.assertEqual(
                resumed.manifest.stages["active"].status,
                StageStatus.PENDING,
            )
            self.assertEqual(resumed.manifest.stages["active"].attempts, 1)
            self.assertIsNone(resumed.manifest.stages["active"].pid)

            persisted = json.loads(
                (run_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(persisted["stages"]["done"]["status"], "pending")
            self.assertFalse(list(run_dir.glob(".manifest.json.*.tmp")))

    def test_resume_preserves_stages_from_other_schedules(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            first = StateStore(run_dir, ("bge",), resume=False)
            first.transition("bge", StageStatus.SUCCEEDED, return_code=0)

            resumed = StateStore(run_dir, ("elser",), resume=True)

            self.assertEqual(
                resumed.manifest.stages["bge"].status,
                StageStatus.SUCCEEDED,
            )
            self.assertEqual(
                resumed.manifest.stages["elser"].status,
                StageStatus.PENDING,
            )

    def test_events_are_valid_append_only_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            store = StateStore(run_dir, ("one",), resume=False)
            store.transition("one", StageStatus.RUNNING, pid=42)
            store.transition("one", StageStatus.SUCCEEDED, return_code=0)

            events = [
                json.loads(line)
                for line in (run_dir / "events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                [event["event"] for event in events],
                ["run_created", "stage_running", "stage_succeeded"],
            )
            self.assertEqual(events[-1]["stage"], "one")


if __name__ == "__main__":
    unittest.main()

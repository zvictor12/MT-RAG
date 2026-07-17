import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

from mtrag.runtime.scheduler import (
    ResourceRequest,
    StageSpec,
    SubprocessScheduler,
    _ResourcePool,
)
from mtrag.runtime.state import StageStatus


def python_command(source: str, *arguments: Path) -> tuple[str, ...]:
    return (sys.executable, "-c", source, *(str(path) for path in arguments))


class ResourcePoolTest(unittest.TestCase):
    def test_gpu_is_exclusive_and_cpu_slots_are_counted(self) -> None:
        pool = _ResourcePool(cpu_slots=3)
        gpu = ResourceRequest(cpu_slots=2, gpu=True)
        other_gpu = ResourceRequest(cpu_slots=1, gpu=True)
        cpu = ResourceRequest(cpu_slots=1)

        self.assertTrue(pool.try_acquire(gpu))
        self.assertFalse(pool.try_acquire(other_gpu))
        self.assertTrue(pool.try_acquire(cpu))
        self.assertFalse(pool.try_acquire(cpu))
        pool.release(cpu)
        pool.release(gpu)
        self.assertTrue(pool.try_acquire(other_gpu))


class SchedulerTest(unittest.TestCase):
    def test_dependencies_condition_and_logs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "order.json"
            write = (
                "import json, pathlib, sys; "
                "path = pathlib.Path(sys.argv[1]); "
                "items = json.loads(path.read_text()) if path.exists() else []; "
                "items.append(sys.argv[2]); "
                "path.write_text(json.dumps(items))"
            )
            stages = (
                StageSpec("first", python_command(write, output) + ("first",)),
                StageSpec(
                    "optional",
                    python_command(write, output) + ("optional",),
                    dependencies=("first",),
                    condition=lambda _manifest: False,
                ),
                StageSpec(
                    "last",
                    python_command(write, output) + ("last",),
                    dependencies=("optional",),
                ),
            )

            manifest = SubprocessScheduler(
                stages,
                root / "run",
                cpu_slots=2,
                resume=False,
            ).run_sync()

            self.assertTrue(manifest.complete)
            self.assertEqual(json.loads(output.read_text()), ["first", "last"])
            self.assertEqual(
                manifest.stages["optional"].status,
                StageStatus.SKIPPED,
            )
            self.assertTrue((root / "run/logs/first.log").exists())

    def test_failure_blocks_dependent_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stages = (
                StageSpec("bad", python_command("raise SystemExit(3)")),
                StageSpec(
                    "child",
                    python_command("raise SystemExit(0)"),
                    dependencies=("bad",),
                ),
            )
            manifest = SubprocessScheduler(
                stages,
                root,
                resume=False,
            ).run_sync()
            self.assertEqual(manifest.stages["bad"].status, StageStatus.FAILED)
            self.assertEqual(
                manifest.stages["child"].status,
                StageStatus.BLOCKED,
            )

    def test_resume_does_not_repeat_successful_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            counter = root / "counter.txt"
            count = (
                "import pathlib, sys; p=pathlib.Path(sys.argv[1]); "
                "n=int(p.read_text()) if p.exists() else 0; "
                "p.write_text(str(n+1))"
            )
            first_run = (
                StageSpec("once", python_command(count, counter)),
                StageSpec(
                    "retry_later",
                    python_command("raise SystemExit(4)"),
                    dependencies=("once",),
                ),
            )
            first = SubprocessScheduler(
                first_run,
                root / "run",
                resume=False,
            ).run_sync()
            self.assertEqual(first.stages["retry_later"].status, StageStatus.FAILED)

            second_run = (
                StageSpec("once", python_command(count, counter)),
                StageSpec(
                    "retry_later",
                    python_command("raise SystemExit(0)"),
                    dependencies=("once",),
                ),
            )
            second = SubprocessScheduler(
                second_run,
                root / "run",
                resume=True,
            ).run_sync()

            self.assertEqual(counter.read_text(), "1")
            self.assertTrue(second.complete)
            self.assertEqual(second.stages["retry_later"].attempts, 2)

    def test_graceful_stop_is_resumable(self) -> None:
        async def scenario(root: Path) -> None:
            source = (
                "import signal, sys, time; "
                "signal.signal(signal.SIGINT, lambda *_: sys.exit(0)); "
                "time.sleep(10)"
            )
            scheduler = SubprocessScheduler(
                (StageSpec("long", python_command(source)),),
                root,
                resume=False,
            )
            running = asyncio.create_task(scheduler.run())
            for _ in range(100):
                if scheduler.manifest.stages["long"].status is StageStatus.RUNNING:
                    break
                await asyncio.sleep(0.01)
            scheduler.request_stop()
            manifest = await asyncio.wait_for(running, timeout=2)
            self.assertEqual(
                manifest.stages["long"].status,
                StageStatus.INTERRUPTED,
            )

        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(scenario(Path(directory)))

    def test_graph_validation(self) -> None:
        command = python_command("raise SystemExit(0)")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "unknown dependencies"):
                SubprocessScheduler(
                    (StageSpec("one", command, dependencies=("missing",)),),
                    Path(directory),
                )
            with self.assertRaisesRegex(ValueError, "dependency cycle"):
                SubprocessScheduler(
                    (
                        StageSpec("one", command, dependencies=("two",)),
                        StageSpec("two", command, dependencies=("one",)),
                    ),
                    Path(directory),
                )


if __name__ == "__main__":
    unittest.main()

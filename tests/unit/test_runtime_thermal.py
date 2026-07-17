import tempfile
import unittest
from pathlib import Path

from mtrag.runtime.thermal import (
    ThermalGuard,
    ThermalThresholds,
    read_cpu_temperature,
)


class SequenceReader:
    def __init__(self, values: list[float | None]) -> None:
        self.values = values
        self.index = 0

    def __call__(self) -> float | None:
        value = self.values[min(self.index, len(self.values) - 1)]
        self.index += 1
        return value


class ThermalGuardTest(unittest.TestCase):
    def test_hot_samples_pause_until_resume_threshold(self) -> None:
        reader = SequenceReader([81.0, 82.0, 75.0, 71.0])
        sleeps: list[float] = []
        guard = ThermalGuard(
            gpu_reader=reader,
            cpu_reader=lambda: None,
            poll_interval=1.0,
            resume_hold=0.0,
            hot_samples=2,
            sleeper=sleeps.append,
        )

        with self.assertLogs("mtrag.runtime.thermal", level="WARNING"):
            guard.wait("gpu")

        self.assertEqual(reader.index, 4)
        self.assertEqual(sleeps, [1.0, 1.0, 1.0])

    def test_temperature_above_the_old_abort_limit_only_pauses(self) -> None:
        reader = SequenceReader([100.0, 71.0])
        sleeps: list[float] = []
        guard = ThermalGuard(
            gpu_reader=reader,
            cpu_reader=lambda: None,
            poll_interval=1.0,
            resume_hold=0.0,
            sleeper=sleeps.append,
        )
        with self.assertLogs("mtrag.runtime.thermal", level="WARNING"):
            guard.wait("gpu")

        self.assertEqual(reader.index, 2)
        self.assertEqual(sleeps, [1.0])

    def test_missing_sensors_warn_once_and_do_not_fail(self) -> None:
        guard = ThermalGuard(
            gpu_reader=lambda: None,
            cpu_reader=lambda: None,
            sleeper=lambda _seconds: None,
        )
        with self.assertLogs("mtrag.runtime.thermal", level="WARNING") as logs:
            guard.wait("gpu")
            guard.wait("gpu")
        self.assertEqual(len(logs.output), 2)

    def test_reads_hottest_coretemp_value_from_sysfs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hwmon = root / "hwmon0"
            hwmon.mkdir()
            (hwmon / "name").write_text("coretemp\n", encoding="utf-8")
            (hwmon / "temp1_input").write_text("65000\n", encoding="utf-8")
            (hwmon / "temp2_input").write_text("72500\n", encoding="utf-8")
            self.assertEqual(read_cpu_temperature(root), 72.5)

    def test_threshold_order_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            ThermalThresholds(pause=80, resume=80, abort=86)


if __name__ == "__main__":
    unittest.main()

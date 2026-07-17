from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


LOGGER = logging.getLogger(__name__)
TemperatureReader = Callable[[], float | None]


@dataclass(frozen=True)
class ThermalThresholds:
    pause: float
    resume: float

    def __post_init__(self) -> None:
        if not self.resume < self.pause:
            raise ValueError("thermal thresholds must satisfy resume < pause")


@dataclass(frozen=True)
class ThermalSample:
    gpu_celsius: float | None
    cpu_celsius: float | None


def read_nvidia_temperature() -> float | None:
    """Return the hottest NVIDIA GPU temperature reported by nvidia-smi."""

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None

    temperatures: list[float] = []
    for line in result.stdout.splitlines():
        try:
            temperatures.append(float(line.strip()))
        except ValueError:
            continue
    return max(temperatures, default=None)


def _valid_temperature(value: float) -> bool:
    return 0.0 < value < 150.0


def _read_hwmon_cpu_temperature(root: Path) -> float | None:
    temperatures: list[float] = []
    supported = {"coretemp", "k10temp", "zenpower", "cpu_thermal"}
    for directory in root.glob("hwmon*"):
        try:
            name = (directory / "name").read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if name.lower() not in supported:
            continue
        for path in directory.glob("temp*_input"):
            try:
                temperature = float(path.read_text(encoding="utf-8")) / 1000.0
            except (OSError, ValueError):
                continue
            if _valid_temperature(temperature):
                temperatures.append(temperature)
    return max(temperatures, default=None)


def _nested_temperatures(value: object) -> Iterable[float]:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.endswith("_input") and isinstance(child, (int, float)):
                temperature = float(child)
                if _valid_temperature(temperature):
                    yield temperature
            else:
                yield from _nested_temperatures(child)


def _read_sensors_cpu_temperature() -> float | None:
    try:
        result = subprocess.run(
            ["sensors", "-j"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None

    supported = ("coretemp", "k10temp", "zenpower", "cpu_thermal")
    temperatures: list[float] = []
    for adapter, values in payload.items():
        if any(name in adapter.lower() for name in supported):
            temperatures.extend(_nested_temperatures(values))
    return max(temperatures, default=None)


def read_cpu_temperature(
    hwmon_root: Path = Path("/sys/class/hwmon"),
) -> float | None:
    """Read a CPU package temperature from hwmon, then lm-sensors."""

    temperature = _read_hwmon_cpu_temperature(hwmon_root)
    if temperature is not None:
        return temperature
    return _read_sensors_cpu_temperature()


class ThermalGuard:
    """Pause batch boundaries until CPU and GPU return to safe temperatures."""

    def __init__(
        self,
        *,
        gpu: ThermalThresholds = ThermalThresholds(80.0, 72.0),
        cpu: ThermalThresholds = ThermalThresholds(90.0, 80.0),
        poll_interval: float = 5.0,
        resume_hold: float = 30.0,
        hot_samples: int = 1,
        gpu_reader: TemperatureReader = read_nvidia_temperature,
        cpu_reader: TemperatureReader = read_cpu_temperature,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        if resume_hold < 0:
            raise ValueError("resume_hold cannot be negative")
        if hot_samples < 1:
            raise ValueError("hot_samples must be at least one")
        self.thresholds = {"gpu": gpu, "cpu": cpu}
        self.readers = {"gpu": gpu_reader, "cpu": cpu_reader}
        self.poll_interval = poll_interval
        self.resume_hold = resume_hold
        self.hot_samples = hot_samples
        self._sleep = sleeper
        self._clock = clock
        self._warned_missing: set[str] = set()

    def sample(self) -> ThermalSample:
        return ThermalSample(
            gpu_celsius=self.readers["gpu"](),
            cpu_celsius=self.readers["cpu"](),
        )

    def wait(self, resource: str = "gpu") -> None:
        """Block at a batch boundary while monitored hardware is too hot.

        GPU work also watches the CPU because tokenization and data transfer can
        keep it busy. CPU-only work does not require a working NVIDIA sensor.
        """

        if resource not in {"gpu", "cpu", "all"}:
            raise ValueError("resource must be 'gpu', 'cpu', or 'all'")
        monitored = ("cpu",) if resource == "cpu" else ("gpu", "cpu")
        hot_counts = {name: 0 for name in monitored}
        paused = False
        cool_since: float | None = None

        while True:
            values = {name: self.readers[name]() for name in monitored}
            available = {
                name: temperature
                for name, temperature in values.items()
                if temperature is not None
            }
            for name in monitored:
                if values[name] is None and name not in self._warned_missing:
                    LOGGER.warning(
                        "%s temperature sensor is unavailable; "
                        "thermal protection for it is disabled",
                        name.upper(),
                    )
                    self._warned_missing.add(name)
            if not available:
                return

            for name, temperature in available.items():
                limit = self.thresholds[name]
                if temperature >= limit.pause:
                    hot_counts[name] += 1
                elif not paused:
                    hot_counts[name] = 0

            if not paused:
                paused = any(
                    count >= self.hot_samples for count in hot_counts.values()
                )
                if not paused:
                    if any(hot_counts.values()):
                        self._sleep(self.poll_interval)
                        continue
                    return
                LOGGER.warning(
                    "thermal pause: %s",
                    ", ".join(
                        f"{name.upper()} {temperature:.1f}°C"
                        for name, temperature in available.items()
                    ),
                )

            cooled = all(
                temperature <= self.thresholds[name].resume
                for name, temperature in available.items()
            )
            if cooled:
                if cool_since is None:
                    cool_since = self._clock()
                if self._clock() - cool_since >= self.resume_hold:
                    LOGGER.info("temperatures are safe; resuming work")
                    return
            else:
                cool_since = None
            self._sleep(self.poll_interval)

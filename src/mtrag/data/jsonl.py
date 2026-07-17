import json
import os
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


JsonObject = dict[str, Any]


def iter_jsonl(path: str | Path) -> Iterator[JsonObject]:
    """Yield JSON objects while preserving useful file and line diagnostics."""
    source = Path(path)
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {source}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object in {source}:{line_number}")
            yield value


def read_jsonl(path: str | Path) -> list[JsonObject]:
    return list(iter_jsonl(path))


def write_jsonl(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
) -> None:
    """Atomically replace a JSONL file with the supplied rows."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False))
                stream.write("\n")
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def append_jsonl(path: str | Path, row: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False))
        stream.write("\n")

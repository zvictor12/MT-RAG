import hashlib
import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def stable_key(*parts: Any) -> str:
    payload = json.dumps(
        parts,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SqliteCache:
    """Small persistent memo cache for expensive deterministic model calls."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                namespace TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (namespace, key)
            )
            """
        )

    def get(self, namespace: str, key: str) -> Any | None:
        row = self.connection.execute(
            "SELECT value FROM cache WHERE namespace = ? AND key = ?",
            (namespace, key),
        ).fetchone()
        return None if row is None else json.loads(row[0])

    def put(self, namespace: str, key: str, value: Any) -> None:
        self.put_many(namespace, {key: value})

    def put_many(self, namespace: str, values: Mapping[str, Any]) -> None:
        rows = [
            (
                namespace,
                key,
                json.dumps(value, ensure_ascii=False, separators=(",", ":")),
            )
            for key, value in values.items()
        ]
        if not rows:
            return
        with self.connection:
            self.connection.executemany(
                """
                INSERT INTO cache(namespace, key, value)
                VALUES (?, ?, ?)
                ON CONFLICT(namespace, key) DO UPDATE SET value = excluded.value
                """,
                rows,
            )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "SqliteCache":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

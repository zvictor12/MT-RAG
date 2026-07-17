import tempfile
import unittest
from pathlib import Path

from mtrag.runtime.cache import SqliteCache, stable_key


class CacheTest(unittest.TestCase):
    def test_round_trip_and_stable_key(self) -> None:
        self.assertEqual(stable_key({"b": 2, "a": 1}), stable_key({"a": 1, "b": 2}))
        with tempfile.TemporaryDirectory() as directory:
            with SqliteCache(Path(directory) / "cache.sqlite") as cache:
                self.assertIsNone(cache.get("test", "key"))
                cache.put("test", "key", {"value": 1})
                cache.put_many(
                    "test",
                    {"second": 4, "third": {"nested": True}},
                )
                self.assertEqual(cache.get("test", "key"), {"value": 1})
                self.assertEqual(cache.get("test", "second"), 4)
                self.assertEqual(
                    cache.get("test", "third"),
                    {"nested": True},
                )


if __name__ == "__main__":
    unittest.main()

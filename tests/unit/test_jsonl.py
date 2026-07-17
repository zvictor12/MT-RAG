import tempfile
import unittest
from pathlib import Path

from mtrag.data.jsonl import append_jsonl, read_jsonl, write_jsonl


class JsonlTests(unittest.TestCase):
    def test_write_read_and_append_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "rows.jsonl"
            write_jsonl(path, [{"text": "Привет"}, {"number": 2}])
            append_jsonl(path, {"done": True})

            self.assertEqual(
                read_jsonl(path),
                [{"text": "Привет"}, {"number": 2}, {"done": True}],
            )

    def test_decode_error_contains_path_and_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.jsonl"
            path.write_text('{"ok": true}\nnot-json\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, rf"{path}:2"):
                read_jsonl(path)


if __name__ == "__main__":
    unittest.main()

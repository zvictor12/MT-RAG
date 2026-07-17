import math
import unittest
from pathlib import Path

from mtrag.experiments.fingerprint import fingerprint


class FingerprintTest(unittest.TestCase):
    def test_mapping_order_does_not_change_the_fingerprint(self) -> None:
        self.assertEqual(
            fingerprint("rewrite", {"temperature": 0.2, "prompt": "abc"}),
            fingerprint("rewrite", {"prompt": "abc", "temperature": 0.2}),
        )

    def test_semantic_input_changes_the_fingerprint(self) -> None:
        baseline = fingerprint(
            "rewrite",
            {"temperature": 0.2, "prompt_sha256": "abc"},
        )

        self.assertEqual(
            baseline,
            "a68ba2d8e994e198367bfa7b6f4567bed4e0a85f8f8f1233619c9a33734fc750",
        )
        self.assertNotEqual(
            baseline,
            fingerprint(
                "rewrite",
                {"temperature": 0.3, "prompt_sha256": "abc"},
            ),
        )
        self.assertNotEqual(
            baseline,
            fingerprint(
                "generate",
                {"temperature": 0.2, "prompt_sha256": "abc"},
            ),
        )

    def test_non_json_inputs_are_rejected(self) -> None:
        with self.assertRaises(TypeError):
            fingerprint("rewrite", {"path": Path("prompt.txt")})
        with self.assertRaises(ValueError):
            fingerprint("rewrite", {"temperature": math.nan})


if __name__ == "__main__":
    unittest.main()

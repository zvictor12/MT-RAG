import unittest
from pathlib import Path

from mtrag.encoding import BGEM3QueryEncoder


class CountingGuard:
    def __init__(self) -> None:
        self.calls = 0

    def wait(self, resource="gpu") -> None:
        self.calls += 1


class FakeModel:
    def __init__(self) -> None:
        self.calls = []

    def encode_queries(self, texts, *, batch_size, **_kwargs):
        self.calls.append((len(texts), batch_size))
        return {
            "dense_vecs": [[1.0, 0.0] for _ in texts],
            "lexical_weights": [{"10": 0.5} for _ in texts],
        }


class BgeQueryEncoderTest(unittest.TestCase):
    def test_thermal_boundaries_are_larger_than_gpu_micro_batches(self) -> None:
        guard = CountingGuard()
        model = FakeModel()
        encoder = BGEM3QueryEncoder(
            Path("unused"),
            batch_size=32,
            guard_chunk_size=256,
            guard=guard,
        )
        encoder._model = model

        features = encoder.encode([f"query {index}" for index in range(300)])

        self.assertEqual(len(features), 300)
        self.assertEqual(model.calls, [(256, 32), (44, 32)])
        self.assertEqual(guard.calls, 2)

    def test_empty_input_does_not_load_the_model(self) -> None:
        encoder = BGEM3QueryEncoder(Path("missing"))
        self.assertEqual(encoder.encode([]), [])
        self.assertIsNone(encoder._model)


if __name__ == "__main__":
    unittest.main()

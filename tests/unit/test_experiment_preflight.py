import tempfile
import unittest
from pathlib import Path

from mtrag.experiments.preflight import _require_files, _validate_bge_mapping


class ExperimentPreflightTest(unittest.TestCase):
    def test_requires_every_pinned_model_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").touch()
            with self.assertRaisesRegex(RuntimeError, "model.bin"):
                _require_files("model", root, ("config.json", "model.bin"))

    def test_dense_mapping_must_match_the_downloaded_query_encoder(self) -> None:
        index = "mtrag-cloud-bge-m3-dense"
        mapping = {
            index: {
                "mappings": {
                    "properties": {
                        "embedding": {
                            "type": "dense_vector",
                            "dims": 1024,
                            "similarity": "dot_product",
                            "index_options": {"type": "int8_hnsw"},
                        }
                    }
                }
            }
        }
        _validate_bge_mapping(index, mapping)

        mapping[index]["mappings"]["properties"]["embedding"]["dims"] = 768
        with self.assertRaisesRegex(RuntimeError, "incompatible mapping"):
            _validate_bge_mapping(index, mapping)


if __name__ == "__main__":
    unittest.main()

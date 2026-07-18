import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from mtrag.experiments.preflight import (
    _check_elser_endpoint,
    _require_files,
    _validate_bge_mapping,
    requirements_for,
)


class ExperimentPreflightTest(unittest.TestCase):
    @patch("mtrag.experiments.preflight.requests.post")
    def test_elser_endpoint_must_run_inference(self, post: Mock) -> None:
        post.return_value.ok = False
        config = SimpleNamespace(
            services=SimpleNamespace(
                elasticsearch_url="http://localhost:9200",
                elser_inference_id="mtrag-elser",
            )
        )

        self.assertFalse(_check_elser_endpoint(config))
        post.assert_called_once_with(
            "http://localhost:9200/_inference/sparse_embedding/mtrag-elser",
            json={"input": "preflight query"},
            timeout=120,
        )

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

    def test_requirements_are_derived_from_the_selected_schedule(self) -> None:
        bge = requirements_for(
            (
                SimpleNamespace(kind="encode", params={}),
                SimpleNamespace(kind="retrieve", params={"method": "dense"}),
            )
        )
        elser = requirements_for(
            (
                SimpleNamespace(
                    kind="retrieve",
                    params={"method": "elser"},
                ),
            )
        )

        self.assertTrue(bge.bge_model)
        self.assertTrue(bge.cuda)
        self.assertEqual(bge.bge_modes, {"dense"})
        self.assertFalse(bge.elser)
        self.assertFalse(bge.reranker)
        self.assertFalse(bge.ollama)

        self.assertTrue(elser.elser)
        self.assertFalse(elser.bge_model)
        self.assertFalse(elser.bge_modes)
        self.assertFalse(elser.cuda)


if __name__ == "__main__":
    unittest.main()

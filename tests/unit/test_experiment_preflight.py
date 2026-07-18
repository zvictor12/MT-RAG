import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from mtrag.experiments.preflight import (
    PreflightError,
    _elser_ready,
    _mapping_issue,
    preflight,
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

        self.assertFalse(_elser_ready(config))
        post.assert_called_once_with(
            "http://localhost:9200/_inference/sparse_embedding/mtrag-elser",
            json={"input": "preflight query"},
            timeout=120,
        )

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
        self.assertIsNone(_mapping_issue(index, mapping))

        mapping[index]["mappings"]["properties"]["embedding"]["dims"] = 768
        self.assertIn("incompatible mapping", _mapping_issue(index, mapping) or "")

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

        self.assertTrue(bge.cuda)
        self.assertEqual(bge.bge_modes, {"dense"})
        self.assertFalse(bge.elser)
        self.assertFalse(bge.ollama)

        self.assertTrue(elser.elser)
        self.assertFalse(elser.bge_modes)
        self.assertFalse(elser.cuda)

    @patch("mtrag.experiments.preflight.ollama_client")
    def test_public_boundary_aggregates_reproducibility_errors(
        self,
        client_factory: Mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = SimpleNamespace(
                run=SimpleNamespace(benchmark_root=Path(directory)),
                models=SimpleNamespace(
                    ollama_model="qwen",
                    ollama_digest="expected-digest",
                ),
            )
            client_factory.return_value.installed_model_digests.return_value = {
                "qwen": "different-digest"
            }
            stage = SimpleNamespace(kind="rewrite", params={})

            with self.assertRaises(PreflightError) as failure:
                preflight(config, stages=(stage,))

            message = str(failure.exception)
            self.assertIn("benchmark is missing", message)
            self.assertIn("Ollama digest changed", message)


if __name__ == "__main__":
    unittest.main()

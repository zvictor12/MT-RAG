import unittest
from unittest.mock import ANY, Mock, call, patch

from scripts import setup_elasticsearch


class SetupElasticsearchTest(unittest.TestCase):
    @patch.object(setup_elasticsearch, "es_request")
    @patch.object(setup_elasticsearch.requests, "post")
    @patch.object(setup_elasticsearch.requests, "get")
    def test_ready_elser_endpoint_is_reused(
        self,
        get: Mock,
        post: Mock,
        es_request: Mock,
    ) -> None:
        get.return_value.ok = True
        post.return_value.ok = True

        setup_elasticsearch.create_elser_endpoint()

        es_request.assert_not_called()

    @patch.object(setup_elasticsearch, "es_request")
    @patch.object(setup_elasticsearch.requests, "post")
    @patch.object(setup_elasticsearch.requests, "get")
    def test_failed_elser_endpoint_is_force_recreated(
        self,
        get: Mock,
        post: Mock,
        es_request: Mock,
    ) -> None:
        get.return_value.ok = True
        post.side_effect = (Mock(ok=False), Mock(ok=True))
        path = "/_inference/sparse_embedding/mtrag-elser"

        setup_elasticsearch.create_elser_endpoint()

        self.assertEqual(
            es_request.call_args_list[:2],
            [
                call("DELETE", path, params={"force": "true"}),
                call(
                    "PUT",
                    path,
                    ANY,
                    timeout=31 * 60,
                    params={"timeout": "30m"},
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()

import unittest

from mtrag.retrieval import DenseRetriever, ElserRetriever, SparseRetriever
from mtrag.schemas import BgeFeatures, SearchQuery


class RecordingGateway:
    def __init__(self) -> None:
        self.searches = []
        self.source = None

    def msearch(self, searches, *, source):
        self.searches.extend(searches)
        self.source = source
        return [[] for _ in searches]


class RetrievalAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.query = SearchQuery(
            task_id="q1",
            domain="cloud",
            text="cloud object storage",
            bge=BgeFeatures(
                dense=(1.0, 0.0),
                sparse={"10": 0.75, "11": 0.0, "12": -1.0},
            ),
        )

    def test_dense_uses_configured_hnsw_candidate_budget(self) -> None:
        gateway = RecordingGateway()
        DenseRetriever(gateway, candidate_multiplier=10).search_many(
            [self.query],
            top_k=50,
        )

        index, body = gateway.searches[0]
        knn = body["query"]["knn"]
        self.assertEqual(index, "mtrag-cloud-bge-m3-dense")
        self.assertEqual(knn["k"], 50)
        self.assertEqual(knn["num_candidates"], 500)
        self.assertEqual(knn["rescore_vector"], {"oversample": 2.0})
        self.assertEqual(gateway.source, "bge_dense")

    def test_sparse_sends_positive_token_weights_without_pruning(self) -> None:
        gateway = RecordingGateway()
        SparseRetriever(gateway).search_many([self.query], top_k=20)

        index, body = gateway.searches[0]
        sparse = body["query"]["sparse_vector"]
        self.assertEqual(index, "mtrag-cloud-bge-m3-sparse")
        self.assertEqual(sparse["query_vector"], {"10": 0.75})
        self.assertFalse(sparse["prune"])

    def test_elser_uses_the_semantic_text_field(self) -> None:
        gateway = RecordingGateway()
        ElserRetriever(gateway).search_many([self.query], top_k=10)

        index, body = gateway.searches[0]
        self.assertEqual(index, "mtrag-cloud-elser")
        self.assertEqual(
            body["query"]["semantic"],
            {"field": "semantic_text", "query": self.query.text},
        )


if __name__ == "__main__":
    unittest.main()

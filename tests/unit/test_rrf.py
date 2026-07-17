import unittest

from mtrag.retrieval.rrf import rrf_fuse
from mtrag.schemas import SearchHit


def hit(document_id: str, rank: int, source: str) -> SearchHit:
    return SearchHit(
        document_id=document_id,
        score=10.0 - rank,
        rank=rank,
        source=source,
        text=document_id,
    )


class RrfTest(unittest.TestCase):
    def test_fuses_by_rank_and_deduplicates_documents(self) -> None:
        results = rrf_fuse(
            {
                "dense": [hit("a", 1, "dense"), hit("b", 2, "dense")],
                "sparse": [hit("b", 1, "sparse"), hit("c", 2, "sparse")],
            },
            rank_constant=60,
            top_k=3,
        )

        self.assertEqual([item.document_id for item in results], ["b", "a", "c"])
        self.assertEqual(results[0].rank, 1)
        self.assertEqual(results[0].components["dense_rank"], 2.0)
        self.assertEqual(results[0].components["sparse_rank"], 1.0)


if __name__ == "__main__":
    unittest.main()

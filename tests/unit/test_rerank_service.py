import tempfile
import unittest
from pathlib import Path

from mtrag.reranking.service import RerankService, passage_text
from mtrag.runtime.cache import SqliteCache
from mtrag.schemas import SearchHit


class FakeScorer:
    def __init__(self) -> None:
        self.calls = 0

    def score(self, pairs):
        self.calls += 1
        return [1.0 if "relevant" in passage else -1.0 for _, passage in pairs]


class RerankServiceTest(unittest.TestCase):
    def test_title_is_not_duplicated_when_corpus_text_already_starts_with_it(self) -> None:
        hit = SearchHit(
            document_id="d",
            score=1.0,
            rank=1,
            source="dense",
            title="French Revolution",
            text="French Revolution\nThe Directory assumed control.",
        )
        self.assertEqual(
            passage_text(hit),
            "French Revolution\nThe Directory assumed control.",
        )

    def test_scores_sorts_and_reuses_cache(self) -> None:
        scorer = FakeScorer()
        candidates = {
            "q1": [
                SearchHit("empty", 10.0, 1, "rrf", text="\n"),
                SearchHit("bad", 9.0, 1, "rrf", text="unrelated"),
                SearchHit("good", 8.0, 2, "rrf", text="relevant passage"),
            ],
            "q2": [SearchHit("also-empty", 7.0, 1, "rrf", title=" ")],
        }
        with tempfile.TemporaryDirectory() as directory:
            with SqliteCache(Path(directory) / "cache.sqlite") as cache:
                service = RerankService(
                    scorer,
                    cache=cache,
                    model_revision="test",
                )
                first = service.rerank_many(
                    {"q1": "query", "q2": "another query"},
                    candidates,
                    top_k=2,
                )
                second = service.rerank_many(
                    {"q1": "query", "q2": "another query"},
                    candidates,
                    top_k=2,
                )

        self.assertEqual([hit.document_id for hit in first["q1"]], ["good", "bad"])
        self.assertEqual(first["q2"], [])
        self.assertEqual(first, second)
        self.assertEqual(scorer.calls, 1)


if __name__ == "__main__":
    unittest.main()

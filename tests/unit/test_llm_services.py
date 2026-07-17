import tempfile
import unittest
from pathlib import Path

from mtrag.llm.generator import AnswerGenerator
from mtrag.llm.rewriter import QueryRewriter
from mtrag.runtime.cache import SqliteCache
from mtrag.schemas import BenchmarkTask, Context, Message


class FakeClient:
    def __init__(self) -> None:
        self.model_name = "fake"
        self.calls = 0
        self.options = []

    def chat(self, messages, *, output_schema=None, options=None):
        del output_schema
        self.calls += 1
        self.options.append(dict(options or {}))
        if "query rewriting assistant" in messages[0]["content"]:
            return "standalone question"
        return "answer"


def task() -> BenchmarkTask:
    return BenchmarkTask(
        task_id="conversation<::>2",
        conversation_id="conversation",
        turn=2,
        collection="mt-rag-clapnq-elser-512-100-20240503",
        domain="clapnq",
        messages=(
            Message("user", "Who founded it?"),
            Message("agent", "Which company?"),
            Message("user", "IBM"),
        ),
    )


class LlmServicesTest(unittest.TestCase):
    def test_rewrite_and_generation_are_cached(self) -> None:
        client = FakeClient()
        with tempfile.TemporaryDirectory() as directory:
            with SqliteCache(Path(directory) / "cache.sqlite") as cache:
                rewriter = QueryRewriter(client, model_name="fake", cache=cache)
                self.assertEqual(rewriter.rewrite(task()), "standalone question")
                self.assertEqual(rewriter.rewrite(task()), "standalone question")

                generator = AnswerGenerator(
                    client,
                    model_name="fake",
                    cache=cache,
                    temperature=0.1,
                )
                contexts = [Context("doc", "IBM was founded in 1911.")]
                self.assertEqual(generator.generate(task(), contexts), "answer")
                self.assertEqual(generator.generate(task(), contexts), "answer")

                another_temperature = AnswerGenerator(
                    client,
                    model_name="fake",
                    cache=cache,
                    temperature=0.2,
                )
                self.assertEqual(
                    another_temperature.generate(task(), contexts),
                    "answer",
                )

        self.assertEqual(client.calls, 3)

    def test_rewrite_temperature_changes_the_cache_key_and_request(self) -> None:
        client = FakeClient()
        with tempfile.TemporaryDirectory() as directory:
            with SqliteCache(Path(directory) / "cache.sqlite") as cache:
                deterministic = QueryRewriter(
                    client,
                    model_name="fake",
                    cache=cache,
                    temperature=0.0,
                )
                exploratory = QueryRewriter(
                    client,
                    model_name="fake",
                    cache=cache,
                    temperature=0.2,
                )

                deterministic.rewrite(task())
                deterministic.rewrite(task())
                exploratory.rewrite(task())
                exploratory.rewrite(task())

        self.assertEqual(client.calls, 2)
        self.assertEqual(
            [options["temperature"] for options in client.options],
            [0.0, 0.2],
        )


if __name__ == "__main__":
    unittest.main()

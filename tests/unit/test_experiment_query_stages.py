import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from mtrag.experiments.artifacts import RunArtifacts, read_jsonl, write_jsonl_atomic
from mtrag.experiments.query_stages import rewrite_qwen, rewrite_qwen_t02
from mtrag.schemas import BenchmarkTask, Message


COLLECTION = "mt-rag-clapnq-elser-512-100-20240503"


def task(
    conversation_id: str,
    turn: int,
    messages: tuple[Message, ...],
) -> BenchmarkTask:
    return BenchmarkTask(
        task_id=f"{conversation_id}<::>{turn}",
        conversation_id=conversation_id,
        turn=turn,
        collection=COLLECTION,
        domain="clapnq",
        messages=messages,
    )


class RewriteStageTest(unittest.TestCase):
    def test_single_turns_bypass_qwen_and_existing_rows_are_repaired(self) -> None:
        first = task(
            "conversation",
            1,
            (Message("user", "Where do the Cardinals play this week?"),),
        )
        second = task(
            "conversation",
            2,
            (
                Message("user", "Where do the Cardinals play this week?"),
                Message("agent", "They play in Arizona."),
                Message("user", "Is it indoors?"),
            ),
        )
        another_first = task(
            "another",
            1,
            (Message("user", "What is Cloudant?"),),
        )

        with tempfile.TemporaryDirectory() as directory:
            artifacts = RunArtifacts(Path(directory))
            write_jsonl_atomic(
                artifacts.qwen_queries,
                [
                    {
                        "task_id": first.task_id,
                        "query": "Arizona Cardinals schedule",
                        "model": "qwen",
                    }
                ],
            )
            repository = MagicMock()
            repository.load_tasks.return_value = (first, second, another_first)
            rewriter = MagicMock()
            rewriter.rewrite.return_value = (
                "Do the Arizona Cardinals play indoors this week?"
            )
            client = MagicMock()
            config = SimpleNamespace(
                run=SimpleNamespace(benchmark_root=Path("benchmark")),
                models=SimpleNamespace(
                    ollama_model="qwen",
                    ollama_digest="digest",
                ),
                rewriting=SimpleNamespace(max_tokens=128),
            )

            with (
                patch(
                    "mtrag.experiments.query_stages.BenchmarkRepository",
                    return_value=repository,
                ),
                patch(
                    "mtrag.experiments.query_stages.ollama_client",
                    return_value=client,
                ),
                patch(
                    "mtrag.experiments.query_stages.QueryRewriter",
                    return_value=rewriter,
                ),
                patch("mtrag.experiments.query_stages.thermal_guard"),
            ):
                rewrite_qwen(config, artifacts)

            records = {
                record["task_id"]: record
                for record in read_jsonl(artifacts.qwen_queries)
            }
            self.assertEqual(records[first.task_id]["query"], first.messages[-1].text)
            self.assertEqual(records[first.task_id]["rewrite_method"], "identity")
            self.assertEqual(
                records[another_first.task_id]["query"],
                another_first.messages[-1].text,
            )
            self.assertEqual(
                records[second.task_id]["query"],
                "Do the Arizona Cardinals play indoors this week?",
            )
            self.assertEqual(records[second.task_id]["rewrite_method"], "qwen")
            self.assertEqual(
                records[second.task_id]["rewrite_version"],
                "qwen-rewrite-v2",
            )
            rewriter.rewrite.assert_called_once_with(second)
            client.unload.assert_called_once_with()

    def test_temperature_variant_uses_a_separate_checkpoint(self) -> None:
        first = task(
            "conversation",
            1,
            (Message("user", "What is Cloudant?"),),
        )
        second = task(
            "conversation",
            2,
            (
                Message("user", "What is Cloudant?"),
                Message("agent", "It is a database."),
                Message("user", "Who operates it?"),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            artifacts = RunArtifacts(Path(directory))
            write_jsonl_atomic(
                artifacts.qwen_queries,
                [
                    {"task_id": first.task_id, "query": first.messages[-1].text},
                    {"task_id": second.task_id, "query": "Who operates Cloudant?"},
                ],
            )
            repository = MagicMock()
            repository.load_tasks.return_value = (first, second)
            rewriter = MagicMock()
            rewriter.rewrite.return_value = "Which company operates Cloudant?"
            client = MagicMock()
            config = SimpleNamespace(
                run=SimpleNamespace(benchmark_root=Path("benchmark")),
                models=SimpleNamespace(
                    ollama_model="qwen",
                    ollama_digest="digest",
                ),
                rewriting=SimpleNamespace(
                    max_tokens=128,
                    variants=(
                        SimpleNamespace(name="qwen_t0", temperature=0.0),
                        SimpleNamespace(name="qwen_t02", temperature=0.2),
                    ),
                ),
            )

            with (
                patch(
                    "mtrag.experiments.query_stages.BenchmarkRepository",
                    return_value=repository,
                ),
                patch(
                    "mtrag.experiments.query_stages.ollama_client",
                    return_value=client,
                ),
                patch(
                    "mtrag.experiments.query_stages.QueryRewriter",
                    return_value=rewriter,
                ) as rewriter_type,
                patch("mtrag.experiments.query_stages.thermal_guard"),
            ):
                rewrite_qwen_t02(config, artifacts)

            records = read_jsonl(artifacts.rewrite_queries("qwen_t02"))
            self.assertEqual([row["temperature"] for row in records], [0.2, 0.2])
            self.assertEqual(records[0]["rewrite_method"], "identity")
            self.assertEqual(records[1]["query"], "Which company operates Cloudant?")
            self.assertEqual(
                rewriter_type.call_args.kwargs["temperature"],
                0.2,
            )


if __name__ == "__main__":
    unittest.main()

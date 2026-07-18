import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from mtrag.data.jsonl import read_jsonl
from mtrag.experiments.artifacts import RunArtifacts
from mtrag.experiments.query_stages import rewrite_query
from mtrag.llm import AgenticRewrite
from mtrag.schemas import ArtifactRef, BenchmarkTask, Message


COLLECTION = "mt-rag-clapnq-elser-512-100-20240503"


def task(turn: int, messages: tuple[Message, ...]) -> BenchmarkTask:
    return BenchmarkTask(
        task_id=f"conversation<::>{turn}",
        conversation_id="conversation",
        turn=turn,
        collection=COLLECTION,
        domain="clapnq",
        messages=messages,
    )


class RewriteStageTest(unittest.TestCase):
    def test_named_rewrite_uses_its_own_prompt_temperature_and_revision(self) -> None:
        first = task(1, (Message("user", "What is Cloudant?"),))
        second = task(
            2,
            (
                Message("user", "What is Cloudant?"),
                Message("agent", "It is a database."),
                Message("user", "Who operates it?"),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt = root / "rewrite.txt"
            prompt.write_text("Rewrite the query.")
            artifacts = RunArtifacts(root / "run")
            query = SimpleNamespace(
                kind="rewrite",
                prompt=prompt,
                temperature=0.35,
                max_tokens=96,
            )
            config = SimpleNamespace(
                query=lambda _name: query,
                run=SimpleNamespace(benchmark_root=root / "benchmark"),
                models=SimpleNamespace(
                    ollama_model="qwen",
                    ollama_digest="digest",
                    ollama_num_ctx=8192,
                    ollama_seed=42,
                ),
            )
            repository = MagicMock()
            repository.load_tasks.return_value = (first, second)
            rewriter = MagicMock()
            rewriter.rewrite.return_value = "Which company operates Cloudant?"
            client = MagicMock()
            revision = "a" * 64
            artifact = ArtifactRef("qwen_alt", revision)

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
                rewrite_query(
                    config,
                    artifacts,
                    query=artifact,
                )

            records = read_jsonl(artifacts.rewrite(artifact))

        self.assertEqual(records[0]["query"], "What is Cloudant?")
        self.assertEqual(records[0]["rewrite_method"], "identity")
        self.assertEqual(records[1]["query"], "Which company operates Cloudant?")
        self.assertEqual(records[1]["temperature"], 0.35)
        self.assertEqual(rewriter_type.call_args.kwargs["max_tokens"], 96)
        self.assertEqual(rewriter_type.call_args.kwargs["prompt"].text, "Rewrite the query.")
        rewriter.rewrite.assert_called_once_with(second)
        client.unload.assert_called_once_with()

    def test_agentic_rewrite_persists_resolution_trace(self) -> None:
        first = task(1, (Message("user", "What is Cloudant?"),))
        second = task(
            2,
            (
                Message("user", "What is Cloudant?"),
                Message("agent", "It is a database."),
                Message("user", "Who operates it?"),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            question_prompt = root / "question.txt"
            answer_prompt = root / "answer.txt"
            compose_prompt = root / "compose.txt"
            question_prompt.write_text("Ask for missing context.")
            answer_prompt.write_text("Answer from history.")
            compose_prompt.write_text("Compose a query.")
            artifacts = RunArtifacts(root / "run")
            query = SimpleNamespace(
                kind="agentic",
                prompt=question_prompt,
                answer_prompt=answer_prompt,
                compose_prompt=compose_prompt,
                temperature=0.0,
                max_tokens=192,
            )
            config = SimpleNamespace(
                query=lambda _name: query,
                run=SimpleNamespace(benchmark_root=root / "benchmark"),
                models=SimpleNamespace(
                    ollama_model="qwen",
                    ollama_digest="digest",
                    ollama_num_ctx=8192,
                    ollama_seed=42,
                ),
            )
            repository = MagicMock()
            repository.load_tasks.return_value = (first, second)
            agent = MagicMock()
            agent.rewrite.return_value = AgenticRewrite(
                "Which company operates Cloudant?",
                "Q1: What does it refer to?",
                "Q1: RESOLVED\nA1: Cloudant\nE1: U1",
                "resolved",
                ("U1",),
                "QUERY: Which company operates Cloudant?",
            )
            client = MagicMock()
            artifact = ArtifactRef("agentic", "b" * 64)

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
                    "mtrag.experiments.query_stages.HistoryQueryAgent",
                    return_value=agent,
                ),
                patch("mtrag.experiments.query_stages.thermal_guard"),
            ):
                rewrite_query(
                    config,
                    artifacts,
                    query=artifact,
                )

            records = read_jsonl(artifacts.rewrite(artifact))

        self.assertEqual(records[0]["rewrite_method"], "identity")
        self.assertNotIn("clarification_questions", records[0])
        self.assertEqual(records[1]["rewrite_method"], "history_agent")
        self.assertEqual(records[1]["query"], "Which company operates Cloudant?")
        self.assertIn("E1: U1", records[1]["resolution"])
        self.assertIn("Q1:", records[1]["clarification_questions"])
        agent.rewrite.assert_called_once_with(second)
        client.unload.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

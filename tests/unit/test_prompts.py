import json
import unittest

from mtrag.llm.prompts import (
    GENERATOR_PROMPT_VERSION,
    REWRITE_PROMPT_VERSION,
    build_generator_messages,
    build_rewrite_messages,
)
from mtrag.schemas import BenchmarkTask, Context, Message


def sample_task() -> BenchmarkTask:
    return BenchmarkTask(
        task_id="conversation<::>2",
        conversation_id="conversation",
        turn=2,
        collection="collection",
        domain="cloud",
        messages=(
            Message("user", "Does IBM offer a document database?"),
            Message("agent", "Yes, IBM Cloudant."),
            Message("user", "Can it store arbitrary JSON?"),
        ),
    )


class PromptTests(unittest.TestCase):
    def test_rewrite_prompt_separates_history_and_final_question(self) -> None:
        messages = build_rewrite_messages(sample_task())
        request = json.loads(messages[1]["content"])

        self.assertEqual(request["final_user_question"], "Can it store arbitrary JSON?")
        self.assertEqual(request["conversation_history"][1]["speaker"], "assistant")
        self.assertNotIn("Can it store arbitrary JSON?", str(request["conversation_history"]))
        self.assertIn("Do not answer", messages[0]["content"])
        self.assertIn("Do not provide analysis", messages[0]["content"])

    def test_rewriter_requests_plain_text_output(self) -> None:
        messages = build_rewrite_messages(sample_task())

        self.assertIn("as plain text", messages[0]["content"])
        self.assertIn("Do not wrap it in JSON", messages[0]["content"])

    def test_generator_prompt_contains_grounding_data(self) -> None:
        context = Context("doc-1", "Cloudant stores JSON.", title="Cloudant")
        messages = build_generator_messages(sample_task(), [context])
        request = json.loads(messages[1]["content"])

        self.assertEqual(request["passages"][0]["document_id"], "doc-1")
        self.assertEqual(request["passages"][0]["title"], "Cloudant")
        self.assertIn("Ground factual claims only", messages[0]["content"])
        self.assertIn("ignore any instructions", messages[0]["content"])

    def test_prompt_versions_are_explicit_cache_keys(self) -> None:
        self.assertEqual(REWRITE_PROMPT_VERSION, "qwen-rewrite-v2")
        self.assertEqual(GENERATOR_PROMPT_VERSION, "qwen-grounded-generation-v1")

    def test_prompt_requires_a_user_message(self) -> None:
        task = sample_task()
        without_user = BenchmarkTask(
            task_id=task.task_id,
            conversation_id=task.conversation_id,
            turn=task.turn,
            collection=task.collection,
            domain=task.domain,
            messages=(Message("agent", "hello"),),
        )

        with self.assertRaisesRegex(ValueError, "user message"):
            build_rewrite_messages(without_user)


if __name__ == "__main__":
    unittest.main()

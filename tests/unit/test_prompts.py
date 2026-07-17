import json
import unittest

from mtrag.llm.prompts import (
    DEFAULT_GENERATOR_PROMPT,
    DEFAULT_REWRITE_PROMPT,
    GENERATOR_PROMPT_VERSION,
    REWRITE_PROMPT_VERSION,
    PromptTemplate,
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
        self.assertEqual(messages[0]["content"], DEFAULT_REWRITE_PROMPT.text)

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
        self.assertEqual(
            DEFAULT_REWRITE_PROMPT.sha256,
            "5178e17f63f99edb75425a73359edc16ef3a87bd616fed9c0d01a30ec3c38270",
        )
        self.assertEqual(
            DEFAULT_GENERATOR_PROMPT.sha256,
            "ca5cb7ebeb5119d43672e0a87935ef1096449d1eda5e6645ef0d176c832d31c0",
        )

    def test_custom_prompt_replaces_only_the_system_message(self) -> None:
        prompt = PromptTemplate("Rewrite without inventing facts.")

        messages = build_rewrite_messages(sample_task(), prompt=prompt)

        self.assertEqual(messages[0]["content"], prompt.text)
        self.assertEqual(
            json.loads(messages[1]["content"])["final_user_question"],
            "Can it store arbitrary JSON?",
        )

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

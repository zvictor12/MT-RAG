import json
import tempfile
import unittest
from pathlib import Path

from mtrag.llm.history_agent import HistoryQueryAgent
from mtrag.llm.prompts import PromptTemplate
from mtrag.runtime.cache import SqliteCache
from mtrag.schemas import BenchmarkTask, Message


QUESTION = json.dumps({"questions": ['What does "it" refer to?']})
ANSWER = json.dumps(
    {
        "answers": [
            {
                "answer": "IBM Cloudant",
                "evidence_ids": ["U1"],
            }
        ],
    }
)
COMPOSITION = json.dumps({"query": "How much does IBM Cloudant cost?"})


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.requests = []

    def chat(self, messages, *, output_schema=None, options=None):
        self.requests.append(
            {
                "messages": messages,
                "output_schema": output_schema,
                "options": options,
            }
        )
        return self.responses.pop(0)

    @property
    def calls(self) -> int:
        return len(self.requests)


def task() -> BenchmarkTask:
    return BenchmarkTask(
        task_id="conversation<::>2",
        conversation_id="conversation",
        turn=2,
        collection="collection",
        domain="cloud",
        messages=(
            Message("user", "Tell me about IBM Cloudant."),
            Message("agent", "IBM Cloudant is a document database."),
            Message("user", "How much does it cost?"),
        ),
    )


class HistoryQueryAgentTest(unittest.TestCase):
    def agent(self, client, cache=None) -> HistoryQueryAgent:
        return HistoryQueryAgent(
            client,
            model_name="qwen",
            question_prompt=PromptTemplate("Ask for missing context."),
            answer_prompt=PromptTemplate("Answer from history."),
            composition_prompt=PromptTemplate("Compose the query."),
            cache=cache,
        )

    def test_three_structured_roles_are_cached(self) -> None:
        client = FakeClient([QUESTION, ANSWER, COMPOSITION])
        with tempfile.TemporaryDirectory() as directory:
            with SqliteCache(Path(directory) / "cache.sqlite") as cache:
                agent = self.agent(client, cache)
                first = agent.rewrite(task())
                second = agent.rewrite(task())

        self.assertEqual(first, second)
        self.assertEqual(first.query, "How much does IBM Cloudant cost?")
        self.assertEqual(first.status, "resolved")
        self.assertEqual(first.evidence_ids, ("U1",))
        self.assertEqual(json.loads(first.questions)["questions"][0], 'What does "it" refer to?')
        self.assertEqual(json.loads(first.resolution)["resolutions"][0]["answer"], "IBM Cloudant")
        self.assertEqual(json.loads(first.composition)["query"], first.query)
        self.assertEqual(client.calls, 3)
        self.assertTrue(all(request["output_schema"] for request in client.requests))

    def test_empty_questions_is_the_only_standalone_result(self) -> None:
        client = FakeClient(['{"questions": []}'])

        result = self.agent(client).rewrite(task())

        self.assertEqual(result.query, "How much does it cost?")
        self.assertEqual(result.status, "standalone")
        self.assertEqual(json.loads(result.questions), {"questions": []})
        self.assertEqual(client.calls, 1)

    def test_multiple_questions_have_matching_grounded_answers(self) -> None:
        questions = json.dumps(
            {"questions": ["What does it refer to?", "Which cost is meant?"]}
        )
        answers = json.dumps(
            {
                "answers": [
                    {"answer": "IBM Cloudant", "evidence_ids": ["U1"]},
                    {"answer": "service price", "evidence_ids": ["A1"]},
                ]
            }
        )
        client = FakeClient([questions, answers, COMPOSITION])

        result = self.agent(client).rewrite(task())

        self.assertEqual(result.evidence_ids, ("U1", "A1"))
        answer_schema = client.requests[1]["output_schema"]
        self.assertEqual(answer_schema["properties"]["answers"]["minItems"], 2)
        self.assertEqual(answer_schema["properties"]["answers"]["maxItems"], 2)
        request = json.loads(client.requests[2]["messages"][1]["content"])
        self.assertEqual(len(request["resolved_dependencies"]), 2)

    def test_invalid_questions_retry_then_raise_without_fallback(self) -> None:
        client = FakeClient(["NONE", '{"questions": [""]}'])

        with self.assertRaisesRegex(RuntimeError, "conversation<::>2.*history_questions"):
            self.agent(client).rewrite(task())

        self.assertEqual(client.calls, 2)
        self.assertIn("Invalid response", client.requests[1]["messages"][-1]["content"])

    def test_empty_answer_is_retried(self) -> None:
        invalid = '{"answers": [{"answer": "", "evidence_ids": []}]}'
        client = FakeClient([QUESTION, invalid, ANSWER, COMPOSITION])

        result = self.agent(client).rewrite(task())

        self.assertEqual(result.status, "resolved")
        self.assertEqual(client.calls, 4)
        self.assertIn("answer is empty", client.requests[2]["messages"][-1]["content"])

    def test_empty_answer_twice_raises_without_fallback(self) -> None:
        invalid = '{"answers": [{"answer": "", "evidence_ids": []}]}'
        client = FakeClient([QUESTION, invalid, invalid])

        with self.assertRaisesRegex(RuntimeError, "conversation<::>2.*history_answers"):
            self.agent(client).rewrite(task())

        self.assertEqual(client.calls, 3)

    def test_unknown_evidence_is_discarded(self) -> None:
        invalid = json.dumps(
            {
                "answers": [
                    {
                        "answer": "IBM Cloudant",
                        "evidence_ids": ["U99"],
                    }
                ],
            }
        )
        client = FakeClient([QUESTION, invalid, COMPOSITION])

        result = self.agent(client).rewrite(task())

        self.assertEqual(result.evidence_ids, ())
        self.assertEqual(client.calls, 3)

    def test_answer_can_be_grounded_paraphrase(self) -> None:
        paraphrase = json.dumps(
            {
                "answers": [
                    {
                        "answer": "the IBM Cloudant database",
                        "evidence_ids": ["A1"],
                    }
                ],
            }
        )
        client = FakeClient([QUESTION, paraphrase, COMPOSITION])

        result = self.agent(client).rewrite(task())

        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.evidence_ids, ("A1",))

    def test_unknown_extra_evidence_is_discarded(self) -> None:
        answer = json.dumps(
            {
                "answers": [
                    {
                        "answer": "IBM Cloudant",
                        "evidence_ids": ["Q1", "U1"],
                    }
                ]
            }
        )
        client = FakeClient([QUESTION, answer, COMPOSITION])

        result = self.agent(client).rewrite(task())

        self.assertEqual(result.evidence_ids, ("U1",))

    def test_composer_receives_raw_cited_history(self) -> None:
        client = FakeClient([QUESTION, ANSWER, COMPOSITION])

        self.agent(client).rewrite(task())

        request = json.loads(client.requests[2]["messages"][1]["content"])
        evidence = request["resolved_dependencies"][0]["evidence"][0]
        self.assertEqual(evidence["id"], "U1")
        self.assertEqual(evidence["text"], "Tell me about IBM Cloudant.")

    def test_invalid_composition_is_retried(self) -> None:
        client = FakeClient(
            [QUESTION, ANSWER, '{"query": ""}', COMPOSITION]
        )

        result = self.agent(client).rewrite(task())

        self.assertEqual(result.query, "How much does IBM Cloudant cost?")
        self.assertEqual(client.calls, 4)

    def test_unchanged_composition_is_valid_model_output(self) -> None:
        unchanged = '{"query": "How much might it cost?"}'
        client = FakeClient([QUESTION, ANSWER, unchanged])

        result = self.agent(client).rewrite(task())

        self.assertEqual(result.query, "How much might it cost?")
        self.assertEqual(client.calls, 3)

    def test_invalid_composition_twice_raises_without_fallback(self) -> None:
        invalid = '{"query": ""}'
        client = FakeClient([QUESTION, ANSWER, invalid, invalid])

        with self.assertRaisesRegex(RuntimeError, "conversation<::>2.*history_composition"):
            self.agent(client).rewrite(task())

        self.assertEqual(client.calls, 4)

    def test_invalid_answers_are_not_cached(self) -> None:
        invalid = '{"answers": [{"answer": "", "evidence_ids": []}]}'
        with tempfile.TemporaryDirectory() as directory:
            with SqliteCache(Path(directory) / "cache.sqlite") as cache:
                first = FakeClient([QUESTION, invalid, invalid])
                with self.assertRaises(RuntimeError):
                    self.agent(first, cache).rewrite(task())

                second = FakeClient([ANSWER, COMPOSITION])
                result = self.agent(second, cache).rewrite(task())

        self.assertEqual(result.status, "resolved")
        self.assertEqual(first.calls, 3)
        self.assertEqual(second.calls, 2)


if __name__ == "__main__":
    unittest.main()

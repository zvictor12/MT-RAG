import unittest

from mtrag.schemas import BgeFeatures, Message, QueryVariant, SearchHit, SearchQuery


class SchemaTests(unittest.TestCase):
    def test_agent_message_maps_to_ollama_assistant_role(self) -> None:
        message = Message(speaker="agent", text="Hello")

        self.assertEqual(
            message.as_chat_message(),
            {"role": "assistant", "content": "Hello"},
        )

    def test_query_variant_is_a_string_enum(self) -> None:
        self.assertEqual(QueryVariant.GOLD, "gold")

    def test_search_query_can_carry_both_bge_features(self) -> None:
        features = BgeFeatures(dense=(0.1, 0.2), sparse={"42": 0.7})
        query = SearchQuery("task<::>1", "cloud", "query", features)

        self.assertIs(query.bge, features)

    def test_search_hit_component_dicts_are_independent(self) -> None:
        first = SearchHit("a", 1.0, 1, "rrf")
        second = SearchHit("b", 0.5, 2, "rrf")

        first.components["dense_rank"] = 1.0

        self.assertEqual(second.components, {})


if __name__ == "__main__":
    unittest.main()

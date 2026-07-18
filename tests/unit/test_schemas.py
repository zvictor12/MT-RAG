import unittest

from mtrag.schemas import (
    ArtifactRef,
    BenchmarkTask,
    BgeFeatures,
    Message,
    QueryVariant,
    SearchHit,
    SearchQuery,
)


class SchemaTests(unittest.TestCase):
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

    def test_artifact_ref_keeps_logical_name_and_revision(self) -> None:
        artifact = ArtifactRef("elser_last.base", "abc")

        self.assertEqual(
            (artifact.name, artifact.revision),
            ("elser_last.base", "abc"),
        )

    def test_task_splits_history_at_final_user_turn(self) -> None:
        task = BenchmarkTask(
            task_id="q<::>2",
            conversation_id="q",
            turn=2,
            collection="collection",
            domain="cloud",
            messages=(
                Message("user", "first"),
                Message("agent", "answer"),
                Message("user", "follow-up"),
                Message("agent", "ignored trailing answer"),
            ),
        )

        self.assertEqual(task.final_question, "follow-up")
        self.assertEqual(task.history, task.messages[:2])


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mtrag.experiments.artifacts import (
    RunArtifacts,
    read_jsonl,
    write_jsonl_atomic,
)
from mtrag.experiments.generation_stages import (
    evaluate_generation_bge,
    generate_task_c_bge,
    generate_task_c_bge_selected,
)
from mtrag.experiments.retrieval_stages import (
    select_bge,
    select_bge_variants,
    select_winner,
)
from mtrag.experiments.stages import STAGES


class BgeOnlyFlowTest(unittest.TestCase):
    def test_select_bge_chooses_the_best_eligible_pipeline(self) -> None:
        cases = (
            (
                False,
                {
                    "bge_dense_qwen": 0.50,
                    "bge_sparse_qwen": 0.30,
                    "bge_rrf_qwen": 0.40,
                },
                "bge_dense_qwen",
            ),
            (
                False,
                {
                    "bge_dense_qwen": 0.30,
                    "bge_sparse_qwen": 0.50,
                    "bge_rrf_qwen": 0.40,
                },
                "bge_sparse_qwen",
            ),
            (
                True,
                {
                    "bge_dense_qwen": 0.30,
                    "bge_sparse_qwen": 0.40,
                    "bge_rrf_qwen_reranked": 0.50,
                },
                "bge_rrf_qwen_reranked",
            ),
        )
        for enabled, scores, expected in cases:
            with (
                self.subTest(enabled=enabled, expected=expected),
                tempfile.TemporaryDirectory() as directory,
            ):
                artifacts = RunArtifacts(Path(directory))
                artifacts.create_directories()
                artifacts.reranker_gate.write_text(
                    json.dumps({"enabled": enabled}),
                    encoding="utf-8",
                )
                for source_name in scores:
                    artifacts.candidates(source_name).write_text(
                        f"candidate:{source_name}\n",
                        encoding="utf-8",
                    )
                    artifacts.prediction(source_name).write_text(
                        f"prediction:{source_name}\n",
                        encoding="utf-8",
                    )

                def evaluation(_config, _artifacts, name):
                    return SimpleNamespace(
                        metrics=SimpleNamespace(ndcg={5: scores[name]})
                    )

                with patch(
                    "mtrag.experiments.retrieval_stages._evaluate_candidate",
                    side_effect=evaluation,
                ) as evaluate:
                    select_bge(SimpleNamespace(), artifacts)

                self.assertEqual(
                    [call.args[2] for call in evaluate.call_args_list],
                    list(scores),
                )
                self.assertEqual(
                    artifacts.candidates("bge_selected").read_text(),
                    f"candidate:{expected}\n",
                )
                self.assertEqual(
                    artifacts.prediction("bge_selected").read_text(),
                    f"prediction:{expected}\n",
                )
                decision = json.loads(artifacts.bge_winner.read_text())
                self.assertEqual(decision["winner"], expected)
                self.assertEqual(decision["score"], scores[expected])
                self.assertEqual(decision["scores"], scores)

    def test_final_selection_compares_elser_with_the_selected_bge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts = RunArtifacts(Path(directory))
            artifacts.create_directories()
            artifacts.reranker_gate.write_text(
                json.dumps({"enabled": False}),
                encoding="utf-8",
            )
            artifacts.rewrite_winner.write_text(
                json.dumps({"winner": "qwen_t02"}),
                encoding="utf-8",
            )
            scores = {"bge_selected": 0.50, "elser_selected": 0.40}
            for source_name in scores:
                artifacts.candidates(source_name).write_text(
                    f"candidate:{source_name}\n",
                    encoding="utf-8",
                )
                artifacts.prediction(source_name).write_text(
                    f"prediction:{source_name}\n",
                    encoding="utf-8",
                )

            def evaluation(_config, _artifacts, name):
                return SimpleNamespace(
                    metrics=SimpleNamespace(ndcg={5: scores[name]})
                )

            with patch(
                "mtrag.experiments.retrieval_stages._evaluate_candidate",
                side_effect=evaluation,
            ) as evaluate:
                select_winner(SimpleNamespace(), artifacts)

            self.assertEqual(
                [call.args[2] for call in evaluate.call_args_list],
                ["bge_selected", "elser_selected"],
            )
            self.assertEqual(
                artifacts.candidates("winner").read_text(),
                "candidate:bge_selected\n",
            )
            decision = json.loads(artifacts.winner.read_text())
            self.assertEqual(decision["winner"], "bge_selected")
            self.assertEqual(decision["rewrite_variant"], "qwen_t02")

    def test_variant_selection_respects_each_reranker_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts = RunArtifacts(Path(directory))
            artifacts.create_directories()
            variants = {
                "last": {"enabled": False},
                "qwen_t0": {"enabled": True},
                "qwen_t02": {"enabled": False},
            }
            artifacts.reranker_variants.write_text(
                json.dumps({"enabled": True, "variants": variants}),
                encoding="utf-8",
            )
            scores = {
                "bge_dense_last": 0.35,
                "bge_sparse_last": 0.31,
                "bge_rrf_last": 0.34,
                "bge_dense_qwen": 0.32,
                "bge_sparse_qwen": 0.30,
                "bge_rrf_qwen": 0.33,
                "bge_rrf_qwen_reranked": 0.50,
                "bge_dense_qwen_t02": 0.40,
                "bge_sparse_qwen_t02": 0.39,
                "bge_rrf_qwen_t02": 0.41,
            }
            for name, score in scores.items():
                artifacts.retrieval_report(name).write_text(
                    json.dumps({"metrics": {"ndcg": {"5": score}}}),
                    encoding="utf-8",
                )
                artifacts.candidates(name).write_text(
                    f"candidate:{name}\n",
                    encoding="utf-8",
                )
                artifacts.prediction(name).write_text(
                    f"prediction:{name}\n",
                    encoding="utf-8",
                )

            select_bge_variants(SimpleNamespace(), artifacts)

            decision = json.loads(artifacts.bge_winner.read_text())
            self.assertEqual(decision["winner"], "bge_rrf_qwen_reranked")
            self.assertEqual(decision["query_variant"], "qwen_t0")
            self.assertNotIn("bge_rrf_last_reranked", decision["scores"])
            self.assertNotIn("bge_rrf_qwen_t02_reranked", decision["scores"])

    def test_task_c_bge_reads_the_provisional_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts = RunArtifacts(Path(directory))
            write_jsonl_atomic(
                artifacts.candidates("bge_selected"),
                [
                    {
                        "task_id": "q<::>1",
                        "contexts": [
                            {
                                "document_id": "empty",
                                "retriever_score": 9.0,
                                "score": 1.0,
                                "rank": 1,
                                "source": "bge_rrf",
                                "title": "",
                                "text": "\n",
                            },
                            {
                                "document_id": "doc-1",
                                "retriever_score": 7.5,
                                "score": 0.5,
                                "rank": 2,
                                "source": "bge_rrf",
                                "title": "Title",
                                "text": "Passage",
                            },
                            {
                                "document_id": "doc-2",
                                "retriever_score": 5.0,
                                "score": 1.0 / 3.0,
                                "rank": 3,
                                "source": "bge_rrf",
                                "text": "Second passage",
                            },
                        ],
                    }
                ],
            )
            config = SimpleNamespace(
                generation=SimpleNamespace(context_top_k=1)
            )

            with patch("mtrag.experiments.generation_stages._generate") as generate:
                generate_task_c_bge(config, artifacts)

            kwargs = generate.call_args.kwargs
            self.assertEqual(kwargs["task_name"], "c_bge")
            self.assertTrue(kwargs["unload_after"])
            contexts = kwargs["contexts_by_task"]["q<::>1"]
            self.assertEqual(len(contexts), 1)
            self.assertEqual(contexts[0].document_id, "doc-1")
            self.assertEqual(contexts[0].text, "Passage")

    def test_bge_generation_evaluation_is_independent_from_final_task_c(self) -> None:
        config = SimpleNamespace()
        artifacts = RunArtifacts(Path("run"))

        with patch(
            "mtrag.experiments.generation_stages._evaluate_generation"
        ) as evaluate:
            evaluate_generation_bge(config, artifacts)

        evaluate.assert_called_once_with(config, artifacts, ("b", "c_bge"))

    def test_selected_task_c_reuses_last_output_with_updated_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts = RunArtifacts(Path(directory))
            artifacts.create_directories()
            artifacts.bge_winner.write_text(
                json.dumps({"winner": "bge_dense_last"}),
                encoding="utf-8",
            )
            write_jsonl_atomic(
                artifacts.generation("c_bge_last"),
                [
                    {
                        "task_id": "q<::>1",
                        "predictions": [{"text": "answer"}],
                        "pipeline": {"task": "c_bge_last"},
                    }
                ],
            )

            with patch(
                "mtrag.experiments.generation_stages._generate_task_c"
            ) as generate:
                generate_task_c_bge_selected(SimpleNamespace(), artifacts)

            generate.assert_not_called()
            record = read_jsonl(artifacts.generation("c_bge_selected"))[0]
            self.assertEqual(record["pipeline"]["task"], "c_bge_selected")

    def test_stage_registry_exposes_the_bge_only_flow(self) -> None:
        self.assertIs(STAGES["select_bge"], select_bge)
        self.assertIs(STAGES["generate_task_c_bge"], generate_task_c_bge)
        self.assertIs(STAGES["evaluate_generation_bge"], evaluate_generation_bge)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

from mtrag.experiments.spec import ExperimentConfig


MINIMAL_CONFIG = """
[run]
name = "pilot"
output_root = "runs"
benchmark_root = "../benchmark"
cpu_slots = 4

[models]
bge_path = "models/bge"
reranker_path = "models/reranker"

[queries.last]
kind = "last_turn"

[pipelines.bge_last]
kind = "bge"
query = "last"

[generators.qwen]
prompt = "prompts/generation.txt"
temperature = 0.1
max_tokens = 256
context_top_k = 5

[generation.task_b]
task = "b"
generator = "qwen"
contexts = "reference"
evaluate = true

[generation.task_c_last]
task = "c"
generator = "qwen"
contexts = "bge_last.dense"
evaluate = true

[schedules.default]
task_a = ["bge_last.dense"]
generation = ["task_b", "task_c_last"]

[retrieval]
bge_index_revision = "bge-index-v1"
elser_index_revision = "elser-index-v1"
rrf_top_k = 20
prediction_top_k = 10

[reranking]
input_top_k = 20
output_top_k = 10

[evaluation]
bertscore_model = "microsoft/deberta-xlarge-mnli"
bertscore_batch_size = 3
"""


def write_config(root: Path, text: str = MINIMAL_CONFIG) -> Path:
    prompt_dir = root / "prompts"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "generation.txt").write_text("Generate an answer.\n")
    path = root / "configs" / "pilot.toml"
    path.parent.mkdir()
    path.write_text(text)
    return path


class ExperimentConfigTest(unittest.TestCase):
    def test_relative_paths_are_resolved_from_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = ExperimentConfig.load(write_config(root))

            self.assertEqual(config.project_root, root)
            self.assertEqual(config.default_run_dir, root / "runs" / "pilot")
            self.assertEqual(config.models.bge_path, root / "models" / "bge")
            self.assertEqual(config.run.benchmark_root, root / "../benchmark")
            self.assertEqual(
                config.generator("qwen").prompt,
                root / "prompts" / "generation.txt",
            )

    def test_official_prediction_limit_is_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text = MINIMAL_CONFIG.replace(
                "prediction_top_k = 10",
                "prediction_top_k = 11",
            )

            with self.assertRaisesRegex(ValueError, "official limit"):
                ExperimentConfig.load(write_config(root, text))

    def test_arbitrary_rewrite_prompt_and_temperature_are_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            custom_prompt = root / "prompts" / "rewrite-custom.txt"
            text = (
                MINIMAL_CONFIG
                + """

[queries.experimental]
kind = "rewrite"
prompt = "prompts/rewrite-custom.txt"
temperature = 0.37
max_tokens = 96

[pipelines.bge_experimental]
kind = "bge"
query = "experimental"
"""
            )
            path = write_config(root, text)
            custom_prompt.write_text("Rewrite the query.\n")

            config = ExperimentConfig.load(path)
            query = config.query("experimental")

            self.assertEqual(query.prompt, custom_prompt)
            self.assertEqual(query.temperature, 0.37)
            self.assertEqual(query.max_tokens, 96)
            self.assertEqual(
                config.pipeline("bge_experimental").query,
                "experimental",
            )

    def test_jobs_and_schedules_reference_named_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = ExperimentConfig.load(write_config(Path(directory)))

            pipeline, output = config.resolve_retrieval_output(
                "bge_last.rrf_reranked"
            )

            self.assertEqual(pipeline.name, "bge_last")
            self.assertEqual(output, "rrf_reranked")
            self.assertEqual(
                config.generation_job("task_c_last").contexts,
                "bge_last.dense",
            )
            self.assertEqual(
                config.schedule("default").generation,
                ("task_b", "task_c_last"),
            )

    def test_pipeline_output_kind_is_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text = MINIMAL_CONFIG.replace(
                'task_a = ["bge_last.dense"]',
                'task_a = ["bge_last.base"]',
            )

            with self.assertRaisesRegex(ValueError, "has no output 'base'"):
                ExperimentConfig.load(write_config(root, text))

    def test_schedule_generation_job_must_exist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text = MINIMAL_CONFIG.replace(
                'generation = ["task_b", "task_c_last"]',
                'generation = ["missing"]',
            )

            with self.assertRaisesRegex(ValueError, "unknown generation job"):
                ExperimentConfig.load(write_config(root, text))

    def test_generation_evaluate_must_be_a_boolean(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text = MINIMAL_CONFIG.replace("evaluate = true", 'evaluate = "false"')

            with self.assertRaisesRegex(ValueError, "must be a boolean"):
                ExperimentConfig.load(write_config(root, text))

    def test_official_bertscore_model_is_fixed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text = MINIMAL_CONFIG.replace(
                "microsoft/deberta-xlarge-mnli",
                "another/model",
            )

            with self.assertRaisesRegex(ValueError, "official IBM evaluator"):
                ExperimentConfig.load(write_config(root, text))

    def test_shared_evaluation_and_reranking_settings_are_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = ExperimentConfig.load(write_config(Path(directory)))

            self.assertEqual(
                config.evaluation.bertscore_model,
                "microsoft/deberta-xlarge-mnli",
            )
            self.assertEqual(config.evaluation.bertscore_batch_size, 3)
            self.assertEqual(config.reranking.input_top_k, 20)
            self.assertFalse(hasattr(config.reranking, "bootstrap_samples"))


if __name__ == "__main__":
    unittest.main()

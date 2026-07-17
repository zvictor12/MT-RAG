import tempfile
import unittest
from pathlib import Path

from mtrag.data.benchmark import DOMAINS
from mtrag.experiments.planning import build_plan, build_workflow
from mtrag.experiments.spec import ExperimentConfig


CONFIG = """
[run]
name = "pilot"
output_root = "runs"
benchmark_root = "benchmark"
cpu_slots = 4

[models]
bge_path = "models/bge"
bge_revision = "bge-rev"
reranker_path = "models/reranker"
reranker_revision = "reranker-rev"
ollama_model = "qwen"
ollama_digest = "qwen-rev"

[queries.last]
kind = "last_turn"

[queries.qwen]
kind = "rewrite"
prompt = "prompts/rewrite.txt"
temperature = {temperature}
max_tokens = 128

[pipelines.bge_last]
kind = "bge"
query = "last"

[pipelines.bge_qwen]
kind = "bge"
query = "qwen"

[pipelines.elser_last]
kind = "elser"
query = "last"

[generators.qwen]
prompt = "prompts/generate.txt"
temperature = 0.1
max_tokens = 256
context_top_k = 5

[generation.answer]
task = "c"
generator = "qwen"
contexts = "bge_qwen.dense"
evaluate = true

[schedules.bge]
task_a = ["bge_last.rrf_reranked", "bge_qwen.dense"]
generation = ["answer"]

[schedules.elser]
task_a = ["elser_last.base"]
generation = []

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
bertscore_batch_size = 2
"""


def fixture(root: Path, temperature: float = 0.2) -> ExperimentConfig:
    config_path = root / "configs" / "experiment.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    (root / "prompts").mkdir(exist_ok=True)
    (root / "prompts/rewrite.txt").write_text("rewrite")
    (root / "prompts/generate.txt").write_text("generate")
    benchmark = root / "benchmark"
    generation = benchmark / "mtrag-human/generation_tasks/reference.jsonl"
    generation.parent.mkdir(parents=True, exist_ok=True)
    generation.write_text("{}\n")
    evaluation = benchmark / "scripts/evaluation"
    evaluation.mkdir(parents=True, exist_ok=True)
    (evaluation / "run_retrieval_eval.py").write_text("# retrieval\n")
    (evaluation / "run_algorithmic.py").write_text("# generation\n")
    (evaluation / "config.yaml").write_text("evaluators: []\n")
    for domain in DOMAINS:
        directory = benchmark / "mtrag-human/retrieval_tasks" / domain
        (directory / "qrels").mkdir(parents=True, exist_ok=True)
        (directory / f"{domain}_lastturn.jsonl").write_text("{}\n")
        (directory / f"{domain}_rewrite.jsonl").write_text("{}\n")
        (directory / "qrels/dev.tsv").write_text("query-id\tcorpus-id\tscore\n")
    config_path.write_text(CONFIG.format(temperature=temperature))
    return ExperimentConfig.load(config_path)


def output_revision(workflow, reference: str) -> str:
    return next(
        stage.params["revision"]
        for stage in workflow.stages
        if stage.kind in {"retrieve", "fuse", "rerank"}
        and stage.params["reference"] == reference
    )


class ExperimentPlanTest(unittest.TestCase):
    def test_schedule_is_the_only_source_of_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workflow = build_workflow(fixture(root), schedule="bge")
            names = [stage.name for stage in workflow.stages]

        self.assertTrue(any(name.startswith("rerank.bge_last") for name in names))
        self.assertTrue(any(name.startswith("generate.answer") for name in names))
        self.assertFalse(any("elser" in name for name in names))
        self.assertFalse(any(word in name for name in names for word in ("decide", "select", "winner")))

    def test_generation_is_grouped_around_one_model_switch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workflow = build_workflow(fixture(Path(directory)), schedule="bge")

        generate = [stage for stage in workflow.stages if stage.kind == "generate"]
        unload = next(
            stage for stage in workflow.stages if stage.kind == "unload_ollama"
        )
        evaluate = next(
            stage
            for stage in workflow.stages
            if stage.kind == "evaluate_generation_batch"
        )

        self.assertEqual(
            unload.dependencies,
            tuple(stage.name for stage in generate),
        )
        self.assertEqual(evaluate.dependencies, (unload.name,))
        self.assertEqual(
            [job["job_name"] for job in evaluate.params["jobs"]],
            ["answer"],
        )

    def test_preflight_is_not_a_cached_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workflow = build_workflow(fixture(Path(directory)), schedule="bge")

        self.assertNotIn("preflight", {stage.kind for stage in workflow.stages})

    def test_unrelated_experiments_keep_their_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = build_workflow(fixture(root, 0.2), schedule="bge")
            second = build_workflow(fixture(root, 0.3), schedule="bge")

        self.assertEqual(
            output_revision(first, "bge_last.rrf_reranked"),
            output_revision(second, "bge_last.rrf_reranked"),
        )
        self.assertNotEqual(
            output_revision(first, "bge_qwen.dense"),
            output_revision(second, "bge_qwen.dense"),
        )

    def test_task_source_changes_only_task_consumers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = fixture(root)
            first = build_workflow(config, schedule="bge")
            generation_tasks = (
                root / "benchmark/mtrag-human/generation_tasks/reference.jsonl"
            )
            generation_tasks.write_text('{"changed": true}\n')
            second = build_workflow(config, schedule="bge")

        self.assertEqual(
            output_revision(first, "bge_last.dense"),
            output_revision(second, "bge_last.dense"),
        )
        self.assertNotEqual(
            next(
                stage.fingerprint
                for stage in first.stages
                if stage.kind == "generate"
            ),
            next(
                stage.fingerprint
                for stage in second.stages
                if stage.kind == "generate"
            ),
        )
        self.assertNotEqual(
            next(
                stage.fingerprint
                for stage in first.stages
                if stage.kind == "evaluate_task_a"
                and stage.params["reference"] == "bge_last.rrf_reranked"
            ),
            next(
                stage.fingerprint
                for stage in second.stages
                if stage.kind == "evaluate_task_a"
                and stage.params["reference"] == "bge_last.rrf_reranked"
            ),
        )

    def test_index_revision_changes_retrieval_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_config = fixture(root)
            first = build_workflow(first_config, schedule="bge")
            config_path = root / "configs/experiment.toml"
            config_path.write_text(
                config_path.read_text().replace("bge-index-v1", "bge-index-v2")
            )
            second = build_workflow(
                ExperimentConfig.load(config_path),
                schedule="bge",
            )

        self.assertNotEqual(
            output_revision(first, "bge_qwen.dense"),
            output_revision(second, "bge_qwen.dense"),
        )

    def test_stage_commands_rebuild_the_named_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = fixture(root)
            run_dir = root / "run"
            plan = build_plan(config, run_dir, schedule="elser")

        for stage in plan:
            self.assertIn(stage.name, stage.command)
            self.assertIn("--schedule", stage.command)
            self.assertIn("elser", stage.command)
            self.assertIn(str(run_dir.resolve()), stage.command)


if __name__ == "__main__":
    unittest.main()

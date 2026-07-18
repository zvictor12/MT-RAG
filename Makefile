-include .env

OLLAMA_MODEL ?= qwen3.5:4b-q4_K_M
MODEL_ROOT ?= $(HOME)/.cache/mtrag/models
BGE_REVISION := 5617a9f61b028005a4858fdac845db406aefb181
RERANKER_REVISION := 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e
EXPERIMENT_CONFIG ?= configs/experiment.toml
EXPERIMENT_SCHEDULE ?= bge
RUN_DIR ?=
RUN_DIR_ARG = $(if $(strip $(RUN_DIR)),--run-dir "$(RUN_DIR)",)

.PHONY: sync sync-ml sync-evaluation sync-experiment es-up es-down es-logs es-setup elser-setup bge-restore models-bge models-reranker models ollama-enable ollama-pull ollama-unload experiment-plan experiment-preflight experiment-run experiment-status experiment-results test

sync:
	uv sync

sync-ml:
	uv sync --extra ml

sync-evaluation:
	uv sync --extra evaluation

sync-experiment:
	uv sync --extra ml --extra evaluation

es-up:
	docker compose up -d elasticsearch

es-down:
	docker compose down

es-logs:
	docker compose logs -f elasticsearch

es-setup:
	uv run python scripts/setup_elasticsearch.py

elser-setup: es-setup

bge-restore:
	uv run python scripts/restore_bge_indices.py

models-bge:
	uv run --extra ml hf download BAAI/bge-m3 config.json pytorch_model.bin colbert_linear.pt sparse_linear.pt sentencepiece.bpe.model special_tokens_map.json tokenizer.json tokenizer_config.json --revision $(BGE_REVISION) --local-dir $(MODEL_ROOT)/bge-m3 --max-workers 4

models-reranker:
	uv run --extra ml hf download BAAI/bge-reranker-v2-m3 config.json model.safetensors sentencepiece.bpe.model special_tokens_map.json tokenizer.json tokenizer_config.json --revision $(RERANKER_REVISION) --local-dir $(MODEL_ROOT)/bge-reranker-v2-m3 --max-workers 4

models: models-bge models-reranker

ollama-enable:
	sudo systemctl enable --now ollama

ollama-pull:
	ollama pull $(OLLAMA_MODEL)

ollama-unload:
	ollama stop $(OLLAMA_MODEL)

experiment-plan:
	uv run --extra ml --extra evaluation python scripts/run_experiment.py plan --schedule "$(EXPERIMENT_SCHEDULE)" --config "$(EXPERIMENT_CONFIG)" $(RUN_DIR_ARG)

experiment-preflight:
	uv run --extra ml --extra evaluation python scripts/run_experiment.py preflight --schedule "$(EXPERIMENT_SCHEDULE)" --config "$(EXPERIMENT_CONFIG)" $(RUN_DIR_ARG)

experiment-run:
	uv run --extra ml --extra evaluation python scripts/run_experiment.py run --schedule "$(EXPERIMENT_SCHEDULE)" --config "$(EXPERIMENT_CONFIG)" $(RUN_DIR_ARG)

experiment-status:
	uv run --extra ml --extra evaluation python scripts/run_experiment.py status --schedule "$(EXPERIMENT_SCHEDULE)" --config "$(EXPERIMENT_CONFIG)" $(RUN_DIR_ARG)

experiment-results:
	uv run --extra ml --extra evaluation python scripts/run_experiment.py results --config "$(EXPERIMENT_CONFIG)" $(RUN_DIR_ARG)

test:
	uv run python -m unittest discover -s tests/unit -v

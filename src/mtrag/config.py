import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    elasticsearch_url: str = os.getenv(
        "ELASTICSEARCH_URL",
        "http://127.0.0.1:9200",
    )
    elser_inference_id: str = os.getenv("ELSER_INFERENCE_ID", "mtrag-elser")
    ollama_url: str = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen3.5:4b-q4_K_M")
    ollama_num_ctx: int = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
    ollama_num_predict: int = int(os.getenv("OLLAMA_NUM_PREDICT", "512"))
    ollama_seed: int = int(os.getenv("OLLAMA_SEED", "42"))
    ollama_keep_alive: str = os.getenv("OLLAMA_KEEP_ALIVE", "10m")
    ollama_timeout: int = int(os.getenv("OLLAMA_TIMEOUT", "600"))
    benchmark_root: Path = Path(
        os.getenv("MTRAG_BENCHMARK_ROOT", "../mt-rag-benchmark")
    ).expanduser()


settings = Settings()

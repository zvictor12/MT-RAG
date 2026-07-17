from mtrag.llm.ollama_client import chat, installed_models
from mtrag.config import settings


models = installed_models()
if settings.ollama_model not in models:
    raise RuntimeError(f"Run: ollama pull {settings.ollama_model}")

response = chat(
    [{"role": "user", "content": "Reply with exactly: OK"}],
    options={"num_predict": 8},
)
print(response)

from mtrag.interfaces import BatchGuard, ChatClient, NoopGuard
from mtrag.llm.prompts import (
    DEFAULT_REWRITE_PROMPT,
    REWRITE_PROMPT_VERSION,
    PromptTemplate,
    build_rewrite_messages,
)
from mtrag.runtime.cache import SqliteCache, stable_key
from mtrag.schemas import BenchmarkTask


class QueryRewriter:
    def __init__(
        self,
        client: ChatClient,
        *,
        model_name: str,
        cache: SqliteCache | None = None,
        guard: BatchGuard | None = None,
        max_tokens: int = 128,
        temperature: float = 0.0,
        prompt: PromptTemplate = DEFAULT_REWRITE_PROMPT,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.cache = cache
        self.guard = guard or NoopGuard()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.prompt = prompt

    def rewrite(self, task: BenchmarkTask) -> str:
        messages = build_rewrite_messages(task, prompt=self.prompt)
        key = stable_key(
            REWRITE_PROMPT_VERSION,
            self.model_name,
            self.max_tokens,
            self.temperature,
            messages,
        )
        cached = self.cache.get("rewrite", key) if self.cache else None
        if cached is not None:
            return str(cached)

        self.guard.wait("gpu")
        query = self.client.chat(
            messages,
            options={
                "num_predict": self.max_tokens,
                "temperature": self.temperature,
            },
        ).strip()
        if not query:
            raise RuntimeError(f"Empty rewrite response for {task.task_id}")
        if self.cache:
            self.cache.put("rewrite", key, query)
        return query

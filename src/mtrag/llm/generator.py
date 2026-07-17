from collections.abc import Sequence

from mtrag.interfaces import BatchGuard, ChatClient, NoopGuard
from mtrag.llm.prompts import (
    DEFAULT_GENERATOR_PROMPT,
    GENERATOR_PROMPT_VERSION,
    PromptTemplate,
    build_generator_messages,
)
from mtrag.runtime.cache import SqliteCache, stable_key
from mtrag.schemas import BenchmarkTask, Context


class AnswerGenerator:
    def __init__(
        self,
        client: ChatClient,
        *,
        model_name: str,
        cache: SqliteCache | None = None,
        guard: BatchGuard | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        prompt: PromptTemplate = DEFAULT_GENERATOR_PROMPT,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.cache = cache
        self.guard = guard or NoopGuard()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.prompt = prompt

    def generate(
        self,
        task: BenchmarkTask,
        contexts: Sequence[Context],
    ) -> str:
        messages = build_generator_messages(task, contexts, prompt=self.prompt)
        key = stable_key(
            GENERATOR_PROMPT_VERSION,
            self.model_name,
            self.max_tokens,
            self.temperature,
            messages,
        )
        cached = self.cache.get("generation", key) if self.cache else None
        if cached is not None:
            return str(cached)

        self.guard.wait("gpu")
        answer = self.client.chat(
            messages,
            options={
                "num_predict": self.max_tokens,
                "temperature": self.temperature,
            },
        ).strip()
        if not answer:
            raise RuntimeError(f"Empty generation response for {task.task_id}")
        if self.cache:
            self.cache.put("generation", key, answer)
        return answer

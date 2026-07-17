from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from mtrag.schemas import BgeFeatures, SearchHit, SearchQuery


class QueryEncoder(Protocol):
    def encode(self, texts: Sequence[str]) -> list[BgeFeatures]: ...


class Retriever(Protocol):
    def search_many(
        self,
        queries: Sequence[SearchQuery],
        *,
        top_k: int,
    ) -> dict[str, list[SearchHit]]: ...


class PairScorer(Protocol):
    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]: ...


class BatchGuard(Protocol):
    def wait(self, resource: str = "gpu") -> None: ...


class ChatClient(Protocol):
    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        output_schema: Mapping[str, Any] | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> str: ...


class NoopGuard:
    def wait(self, resource: str = "gpu") -> None:
        del resource

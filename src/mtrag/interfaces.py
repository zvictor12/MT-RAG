from collections.abc import Mapping, Sequence
from typing import Any, Protocol


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
        pass

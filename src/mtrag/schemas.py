from dataclasses import dataclass, field
from enum import StrEnum


class QueryVariant(StrEnum):
    """Text used as the retrieval query in an experiment."""

    LAST = "last"
    QWEN = "qwen"
    QWEN_T0 = "qwen_t0"
    QWEN_T02 = "qwen_t02"
    GOLD = "gold"


@dataclass(frozen=True, slots=True)
class Message:
    speaker: str
    text: str

    def as_chat_message(self) -> dict[str, str]:
        role = "assistant" if self.speaker == "agent" else self.speaker
        return {"role": role, "content": self.text}


@dataclass(frozen=True, slots=True)
class Context:
    document_id: str
    text: str
    title: str | None = None
    url: str | None = None
    score: float | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkTask:
    task_id: str
    conversation_id: str
    turn: int
    collection: str
    domain: str
    messages: tuple[Message, ...]
    contexts: tuple[Context, ...] = ()
    targets: tuple[Message, ...] = ()
    answerability: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class QueryCase:
    task_id: str
    domain: str
    variant: QueryVariant
    text: str


@dataclass(frozen=True, slots=True)
class BgeFeatures:
    dense: tuple[float, ...]
    sparse: dict[str, float]


@dataclass(frozen=True, slots=True)
class SearchQuery:
    task_id: str
    domain: str
    text: str
    bge: BgeFeatures | None = None


@dataclass(frozen=True, slots=True)
class SearchHit:
    document_id: str
    score: float
    rank: int
    source: str
    title: str | None = None
    text: str | None = None
    components: dict[str, float] = field(default_factory=dict)

    @property
    def has_passage(self) -> bool:
        return bool((self.title or "").strip() or (self.text or "").strip())

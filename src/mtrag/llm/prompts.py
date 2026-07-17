import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from mtrag.schemas import BenchmarkTask, Context, Message


REWRITE_PROMPT_VERSION = "qwen-rewrite-v2"
GENERATOR_PROMPT_VERSION = "qwen-grounded-generation-v1"


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    text: str
    source: Path | None = field(default=None, compare=False)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @classmethod
    def from_file(cls, path: str | Path) -> "PromptTemplate":
        source = Path(path).expanduser().resolve()
        text = source.read_text(encoding="utf-8").removesuffix("\n")
        return cls(text=text, source=source)


_TEMPLATES = Path(__file__).with_name("templates")
DEFAULT_REWRITE_PROMPT = PromptTemplate.from_file(_TEMPLATES / "rewrite-v2.txt")
DEFAULT_GENERATOR_PROMPT = PromptTemplate.from_file(
    _TEMPLATES / "generation-v1.txt"
)


def build_rewrite_messages(
    task: BenchmarkTask,
    *,
    prompt: PromptTemplate = DEFAULT_REWRITE_PROMPT,
) -> list[dict[str, str]]:
    history, question = _history_and_question(task.messages)
    request = {
        "conversation_history": [_message_record(message) for message in history],
        "final_user_question": question,
    }
    return [
        {"role": "system", "content": prompt.text},
        {
            "role": "user",
            "content": json.dumps(request, ensure_ascii=False, indent=2),
        },
    ]


def build_generator_messages(
    task: BenchmarkTask,
    contexts: Sequence[Context],
    *,
    prompt: PromptTemplate = DEFAULT_GENERATOR_PROMPT,
) -> list[dict[str, str]]:
    history, question = _history_and_question(task.messages)
    request = {
        "conversation_history": [_message_record(message) for message in history],
        "final_user_question": question,
        "passages": [_context_record(context) for context in contexts],
    }
    return [
        {"role": "system", "content": prompt.text},
        {
            "role": "user",
            "content": json.dumps(request, ensure_ascii=False, indent=2),
        },
    ]


def _history_and_question(
    messages: Sequence[Message],
) -> tuple[tuple[Message, ...], str]:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].speaker == "user":
            return tuple(messages[:index]), messages[index].text
    raise ValueError("A benchmark task must contain a user message")


def _message_record(message: Message) -> dict[str, str]:
    speaker = "assistant" if message.speaker == "agent" else message.speaker
    return {"speaker": speaker, "text": message.text}


def _context_record(context: Context) -> dict[str, str]:
    record = {
        "document_id": context.document_id,
        "text": context.text,
    }
    if context.title:
        record["title"] = context.title
    return record

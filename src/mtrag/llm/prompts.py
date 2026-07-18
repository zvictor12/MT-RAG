import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from mtrag.schemas import BenchmarkTask, Context, Message


REWRITE_PROMPT_VERSION = "qwen-rewrite-v2"
HISTORY_QUESTION_VERSION = "history-question-v2"
HISTORY_ANSWER_VERSION = "history-answer-v2"
GROUNDED_COMPOSITION_VERSION = "grounded-query-composition-v3"
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
    request = {
        "conversation_history": [
            _message_record(message) for message in task.history
        ],
        "final_user_question": task.final_question,
    }
    return [
        {"role": "system", "content": prompt.text},
        {
            "role": "user",
            "content": json.dumps(request, ensure_ascii=False, indent=2),
        },
    ]


def build_history_question_messages(
    task: BenchmarkTask,
    *,
    prompt: PromptTemplate,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": prompt.text},
        {
            "role": "user",
            "content": json.dumps(
                {"current_question": task.final_question},
                ensure_ascii=False,
                indent=2,
            ),
        },
    ]


def build_history_answer_messages(
    task: BenchmarkTask,
    questions: Sequence[str],
    *,
    prompt: PromptTemplate,
) -> list[dict[str, str]]:
    request = {
        "history": numbered_history(task.history),
        "questions": [
            {"id": f"Q{index}", "text": question}
            for index, question in enumerate(questions, start=1)
        ],
    }
    return [
        {"role": "system", "content": prompt.text},
        {
            "role": "user",
            "content": json.dumps(request, ensure_ascii=False, indent=2),
        },
    ]


def build_grounded_rewrite_messages(
    task: BenchmarkTask,
    dependencies: Sequence[dict[str, str]],
    *,
    prompt: PromptTemplate,
) -> list[dict[str, str]]:
    request = {
        "current_question": task.final_question,
        "resolved_dependencies": list(dependencies),
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
    request = {
        "conversation_history": [
            _message_record(message) for message in task.history
        ],
        "final_user_question": task.final_question,
        "passages": [_context_record(context) for context in contexts],
    }
    return [
        {"role": "system", "content": prompt.text},
        {
            "role": "user",
            "content": json.dumps(request, ensure_ascii=False, indent=2),
        },
    ]


def _message_record(message: Message) -> dict[str, str]:
    speaker = "assistant" if message.speaker == "agent" else message.speaker
    return {"speaker": speaker, "text": message.text}


def numbered_history(messages: Sequence[Message]) -> list[dict[str, str]]:
    counts = {"user": 0, "assistant": 0}
    records = []
    for message in messages:
        speaker = "assistant" if message.speaker == "agent" else message.speaker
        counts[speaker] += 1
        prefix = "U" if speaker == "user" else "A"
        records.append(
            {
                "id": f"{prefix}{counts[speaker]}",
                "speaker": speaker,
                "text": message.text,
            }
        )
    return records


def _context_record(context: Context) -> dict[str, str]:
    record = {
        "document_id": context.document_id,
        "text": context.text,
    }
    if context.title:
        record["title"] = context.title
    return record

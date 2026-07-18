import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

from mtrag.interfaces import BatchGuard, ChatClient, NoopGuard
from mtrag.llm.prompts import (
    GROUNDED_COMPOSITION_VERSION,
    HISTORY_ANSWER_VERSION,
    HISTORY_QUESTION_VERSION,
    PromptTemplate,
    build_grounded_rewrite_messages,
    build_history_answer_messages,
    build_history_question_messages,
    numbered_history,
)
from mtrag.runtime.cache import SqliteCache, stable_key
from mtrag.schemas import BenchmarkTask, Message


AGENT_PROTOCOL_VERSION = "history-agent-json-v3"


def _object_schema(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


QUESTION_SCHEMA = _object_schema(
    {
        "questions": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        }
    }
)
ANSWER_ITEM_SCHEMA = _object_schema(
    {
        "answer": {"type": "string", "minLength": 1},
        "evidence_ids": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 4,
        },
    }
)
COMPOSITION_SCHEMA = _object_schema(
    {"query": {"type": "string", "minLength": 1}}
)

QUESTION_PROTOCOL = (
    "OUTPUT PROTOCOL (replaces the syntax above): return JSON matching the schema. "
    "Put plain question texts in `questions`; use [] only for a standalone query."
)
ANSWER_PROTOCOL = (
    "OUTPUT PROTOCOL (replaces the syntax above): return JSON matching the schema. "
    "Return one answer per question, in the same order, with supporting history IDs."
)
COMPOSITION_PROTOCOL = (
    "OUTPUT PROTOCOL (replaces the syntax above): return JSON matching the schema. "
    "Put only the standalone query in `query`; cited evidence is authoritative."
)


@dataclass(frozen=True, slots=True)
class AgenticRewrite:
    query: str
    questions: str
    resolution: str
    status: str
    evidence_ids: tuple[str, ...] = ()
    composition: str | None = None


@dataclass(frozen=True, slots=True)
class _Dependency:
    question: str
    answer: str
    evidence_ids: tuple[str, ...]


T = TypeVar("T")


class HistoryQueryAgent:
    """Ask what is missing, resolve it from history, then compose a query."""

    def __init__(
        self,
        client: ChatClient,
        *,
        model_name: str,
        question_prompt: PromptTemplate,
        answer_prompt: PromptTemplate,
        composition_prompt: PromptTemplate,
        cache: SqliteCache | None = None,
        guard: BatchGuard | None = None,
        max_tokens: int = 192,
        temperature: float = 0.0,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.question_prompt = question_prompt
        self.answer_prompt = answer_prompt
        self.composition_prompt = composition_prompt
        self.cache = cache
        self.guard = guard or NoopGuard()
        self.max_tokens = max_tokens
        self.temperature = temperature

    def rewrite(self, task: BenchmarkTask) -> AgenticRewrite:
        current = _final_user_question(task)
        questions = self._ask(task)
        questions_trace = _trace({"questions": list(questions)})
        if not questions:
            return AgenticRewrite(current, questions_trace, "", "standalone")

        history = {
            record["id"]: record
            for record in numbered_history(_history_messages(task))
        }
        dependencies = self._answer(task, questions, history)
        resolution_trace = _trace(
            {"resolutions": [_resolution_record(item) for item in dependencies]}
        )
        composition_input = [
            {
                "question": item.question,
                "answer": item.answer,
                "evidence": [
                    history[evidence_id]
                    for evidence_id in item.evidence_ids
                ],
            }
            for item in dependencies
        ]
        query = self._compose(task, composition_input)
        evidence_ids = tuple(
            dict.fromkeys(
                evidence_id
                for item in dependencies
                for evidence_id in item.evidence_ids
            )
        )
        return AgenticRewrite(
            query,
            questions_trace,
            resolution_trace,
            "resolved",
            evidence_ids,
            _trace({"query": query}),
        )

    def _ask(self, task: BenchmarkTask) -> tuple[str, ...]:
        return self._chat_json(
            task.task_id,
            "history_questions",
            HISTORY_QUESTION_VERSION,
            _with_protocol(
                build_history_question_messages(task, prompt=self.question_prompt),
                QUESTION_PROTOCOL,
            ),
            QUESTION_SCHEMA,
            _questions,
            max_tokens=min(192, self.max_tokens),
        )

    def _answer(
        self,
        task: BenchmarkTask,
        questions: tuple[str, ...],
        history: Mapping[str, Mapping[str, str]],
    ) -> tuple[_Dependency, ...]:
        count = len(questions)
        schema = _object_schema(
            {
                "answers": {
                    "type": "array",
                    "items": ANSWER_ITEM_SCHEMA,
                    "minItems": count,
                    "maxItems": count,
                }
            }
        )
        return self._chat_json(
            task.task_id,
            "history_answers",
            HISTORY_ANSWER_VERSION,
            _with_protocol(
                build_history_answer_messages(
                    task, questions, prompt=self.answer_prompt
                ),
                ANSWER_PROTOCOL,
            ),
            schema,
            lambda value: _dependencies(value, questions, history),
            max_tokens=self.max_tokens,
        )

    def _compose(
        self,
        task: BenchmarkTask,
        dependencies: list[dict[str, Any]],
    ) -> str:
        return self._chat_json(
            task.task_id,
            "history_composition",
            GROUNDED_COMPOSITION_VERSION,
            _with_protocol(
                build_grounded_rewrite_messages(
                    task, dependencies, prompt=self.composition_prompt
                ),
                COMPOSITION_PROTOCOL,
            ),
            COMPOSITION_SCHEMA,
            _query,
            max_tokens=min(192, self.max_tokens),
        )

    def _chat_json(
        self,
        task_id: str,
        namespace: str,
        version: str,
        messages: list[dict[str, str]],
        schema: Mapping[str, Any],
        validate: Callable[[Any], T],
        *,
        max_tokens: int,
    ) -> T:
        key = stable_key(
            AGENT_PROTOCOL_VERSION,
            version,
            self.model_name,
            max_tokens,
            self.temperature,
            schema,
            messages,
        )
        cached = self.cache.get(namespace, key) if self.cache else None
        if cached is not None:
            try:
                return validate(cached)
            except (TypeError, ValueError):
                pass

        request = messages
        last_error = "empty response"
        for attempt in range(2):
            self.guard.wait("gpu")
            raw = self.client.chat(
                request,
                output_schema=schema,
                options={
                    "num_predict": max_tokens,
                    "temperature": self.temperature,
                },
            ).strip()
            try:
                value = json.loads(raw)
                result = validate(value)
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                last_error = str(error)
                if attempt == 0:
                    request = [
                        *messages,
                        {"role": "assistant", "content": raw},
                        {
                            "role": "user",
                            "content": (
                                f"Invalid response: {last_error}. "
                                "Return a corrected JSON object matching the schema."
                            ),
                        },
                    ]
                continue

            if self.cache:
                self.cache.put(namespace, key, value)
            return result

        raise RuntimeError(
            f"Agent protocol failed for {task_id} during {namespace}: {last_error}"
        )


def _questions(value: Any) -> tuple[str, ...]:
    if not isinstance(value, dict) or not isinstance(value.get("questions"), list):
        raise ValueError("questions must be an array")
    questions = [
        question.strip()
        for question in value["questions"]
        if isinstance(question, str) and question.strip()
    ]
    if len(questions) != len(value["questions"]):
        raise ValueError("questions must contain only non-empty strings")
    return tuple(questions)


def _dependencies(
    value: Any,
    questions: tuple[str, ...],
    history: Mapping[str, Mapping[str, str]],
) -> tuple[_Dependency, ...]:
    if not isinstance(value, dict) or not isinstance(value.get("answers"), list):
        raise ValueError("answers must be an array")
    answers = value["answers"]
    if len(answers) != len(questions):
        raise ValueError("answer count does not match question count")
    return tuple(
        _dependency(answer, question, history)
        for question, answer in zip(questions, answers, strict=True)
    )


def _dependency(
    value: Any,
    question: str,
    history: Mapping[str, Mapping[str, str]],
) -> _Dependency:
    if not isinstance(value, dict):
        raise ValueError("answer must be an object")
    answer = value.get("answer")
    evidence = value.get("evidence_ids")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("answer is empty")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("answer has no evidence")
    known_evidence = tuple(
        dict.fromkeys(
            item
            for item in evidence
            if isinstance(item, str) and item in history
        )
    )
    return _Dependency(question, answer.strip(), known_evidence)


def _resolution_record(item: _Dependency) -> dict[str, Any]:
    return {
        "status": "resolved",
        "answer": item.answer,
        "evidence_ids": list(item.evidence_ids),
    }


def _query(value: Any) -> str:
    if not isinstance(value, dict) or not isinstance(value.get("query"), str):
        raise ValueError("query must be a string")
    query = value["query"].strip()
    if not query:
        raise ValueError("query must not be empty")
    return query


def _with_protocol(
    messages: list[dict[str, str]],
    protocol: str,
) -> list[dict[str, str]]:
    first, *rest = messages
    return [{**first, "content": f"{first['content']}\n\n{protocol}"}, *rest]


def _trace(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _final_user_question(task: BenchmarkTask) -> str:
    return next(
        message.text
        for message in reversed(task.messages)
        if message.speaker == "user"
    )


def _history_messages(task: BenchmarkTask) -> tuple[Message, ...]:
    for index in range(len(task.messages) - 1, -1, -1):
        if task.messages[index].speaker == "user":
            return task.messages[:index]
    return ()

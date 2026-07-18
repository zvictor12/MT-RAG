import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

from mtrag.interfaces import BatchGuard, ChatClient
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
from mtrag.schemas import BenchmarkTask


AGENT_PROTOCOL_VERSION = "history-agent-json-v3"


def _schema(**properties: Any) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


QUESTION_SCHEMA = _schema(
    questions={
        "type": "array",
        "items": {"type": "string", "minLength": 1},
    }
)
ANSWER_ITEM_SCHEMA = _schema(
    answer={"type": "string", "minLength": 1},
    evidence_ids={
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 4,
    },
)
COMPOSITION_SCHEMA = _schema(query={"type": "string", "minLength": 1})

_PROTOCOL_PREFIX = (
    "OUTPUT PROTOCOL (replaces the syntax above): return JSON matching the schema. "
)
_STEPS = {
    "history_questions": (
        HISTORY_QUESTION_VERSION,
        "Put plain question texts in `questions`; use [] only for a standalone query.",
    ),
    "history_answers": (
        HISTORY_ANSWER_VERSION,
        "Return one answer per question, in the same order, "
        "with supporting history IDs.",
    ),
    "history_composition": (
        GROUNDED_COMPOSITION_VERSION,
        "Put only the standalone query in `query`; cited evidence is authoritative.",
    ),
}


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


@dataclass(slots=True)
class HistoryQueryAgent:
    """Resolve missing query context through three explicit model roles."""

    client: ChatClient
    model_name: str
    question_prompt: PromptTemplate
    answer_prompt: PromptTemplate
    composition_prompt: PromptTemplate
    cache: SqliteCache | None = None
    guard: BatchGuard | None = None
    max_tokens: int = 192
    temperature: float = 0.0

    def rewrite(self, task: BenchmarkTask) -> AgenticRewrite:
        questions = self._chat_json(
            task.task_id,
            "history_questions",
            build_history_question_messages(task, prompt=self.question_prompt),
            QUESTION_SCHEMA,
            lambda response: tuple(
                _required_text(item, "question")
                for item in response["questions"]
            ),
            min(192, self.max_tokens),
        )
        question_trace = _trace({"questions": list(questions)})
        if not questions:
            return AgenticRewrite(
                task.final_question,
                question_trace,
                "",
                "standalone",
            )

        history = {
            record["id"]: record for record in numbered_history(task.history)
        }

        def parse_answers(response: Mapping[str, Any]) -> tuple[_Dependency, ...]:
            return tuple(
                _Dependency(
                    question,
                    _required_text(item["answer"], "answer"),
                    tuple(
                        dict.fromkeys(
                            evidence_id
                            for evidence_id in item["evidence_ids"]
                            if evidence_id in history
                        )
                    ),
                )
                for question, item in zip(
                    questions,
                    response["answers"],
                    strict=True,
                )
            )

        count = len(questions)
        answers_schema = _schema(
            answers={
                "type": "array",
                "items": ANSWER_ITEM_SCHEMA,
                "minItems": count,
                "maxItems": count,
            }
        )
        dependencies = self._chat_json(
            task.task_id,
            "history_answers",
            build_history_answer_messages(
                task,
                questions,
                prompt=self.answer_prompt,
            ),
            answers_schema,
            parse_answers,
            self.max_tokens,
        )

        resolutions = [
            {
                "status": "resolved",
                "answer": item.answer,
                "evidence_ids": list(item.evidence_ids),
            }
            for item in dependencies
        ]
        grounded_dependencies = [
            {
                "question": item.question,
                "answer": item.answer,
                "evidence": [
                    history[evidence_id] for evidence_id in item.evidence_ids
                ],
            }
            for item in dependencies
        ]
        query = self._chat_json(
            task.task_id,
            "history_composition",
            build_grounded_rewrite_messages(
                task,
                grounded_dependencies,
                prompt=self.composition_prompt,
            ),
            COMPOSITION_SCHEMA,
            lambda response: _required_text(response["query"], "query"),
            min(192, self.max_tokens),
        )
        evidence_ids = tuple(
            dict.fromkeys(
                evidence_id
                for item in dependencies
                for evidence_id in item.evidence_ids
            )
        )
        return AgenticRewrite(
            query=query,
            questions=question_trace,
            resolution=_trace({"resolutions": resolutions}),
            status="resolved",
            evidence_ids=evidence_ids,
            composition=_trace({"query": query}),
        )

    def _chat_json(
        self,
        task_id: str,
        namespace: str,
        messages: list[dict[str, str]],
        schema: Mapping[str, Any],
        parse: Callable[[Mapping[str, Any]], T],
        max_tokens: int,
    ) -> T:
        version, protocol = _STEPS[namespace]
        system = messages[0]
        messages[0] = {
            **system,
            "content": f"{system['content']}\n\n{_PROTOCOL_PREFIX}{protocol}",
        }
        key = stable_key(
            AGENT_PROTOCOL_VERSION,
            version,
            self.model_name,
            max_tokens,
            self.temperature,
            schema,
            messages,
        )
        if self.cache and (cached := self.cache.get(namespace, key)) is not None:
            try:
                return parse(cached)
            except (AttributeError, KeyError, TypeError, ValueError):
                pass

        request = messages
        last_error: Exception | str = "empty response"
        for attempt in range(2):
            if self.guard:
                self.guard.wait("gpu")
            raw = self.client.chat(
                request,
                output_schema=schema,
                options={"num_predict": max_tokens, "temperature": self.temperature},
            ).strip()
            try:
                value = json.loads(raw)
                result = parse(value)
            except (AttributeError, KeyError, TypeError, ValueError) as error:
                last_error = error
                if attempt:
                    break
                request = [
                    *messages,
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            f"Invalid response: {error}. Return a corrected JSON "
                            "object matching the schema."
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


def _required_text(value: str, field: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError(f"{field} is empty")
    return text


def _trace(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

from collections.abc import Mapping, Sequence
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from mtrag.config import Settings, settings


def _session() -> requests.Session:
    session = requests.Session()
    session.mount(
        "http://",
        HTTPAdapter(
            max_retries=Retry(
                total=3,
                connect=3,
                read=0,
                status=3,
                backoff_factor=1,
                allowed_methods={"GET", "POST"},
                status_forcelist=(429, 502, 503, 504),
            )
        ),
    )
    return session


class OllamaClient:
    def __init__(self, config: Settings = settings) -> None:
        self.config = config
        self.session = _session()

    @property
    def model_name(self) -> str:
        return self.config.ollama_model

    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        output_schema: Mapping[str, Any] | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> str:
        request_options: dict[str, Any] = {
            "num_ctx": self.config.ollama_num_ctx,
            "num_predict": self.config.ollama_num_predict,
            "temperature": 0,
            "seed": self.config.ollama_seed,
        }
        if options is not None:
            request_options.update(options)

        payload: dict[str, Any] = {
            "model": self.config.ollama_model,
            "messages": list(messages),
            "stream": False,
            "think": False,
            "keep_alive": self.config.ollama_keep_alive,
            "options": request_options,
        }
        if output_schema is not None:
            payload["format"] = output_schema

        response = self.session.post(
            f"{self.config.ollama_url}/api/chat",
            json=payload,
            timeout=self.config.ollama_timeout,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def installed_model_digests(self) -> dict[str, str]:
        response = self.session.get(
            f"{self.config.ollama_url}/api/tags",
            timeout=10,
        )
        response.raise_for_status()
        return {
            str(model["name"]): str(model["digest"])
            for model in response.json()["models"]
        }

    def installed_models(self) -> set[str]:
        return set(self.installed_model_digests())

    def unload(self) -> None:
        response = self.session.post(
            f"{self.config.ollama_url}/api/generate",
            json={
                "model": self.config.ollama_model,
                "prompt": "",
                "stream": False,
                "keep_alive": 0,
            },
            timeout=self.config.ollama_timeout,
        )
        response.raise_for_status()


_default_client = OllamaClient()


def chat(
    messages: Sequence[Mapping[str, str]],
    *,
    output_schema: Mapping[str, Any] | None = None,
    think: bool = False,
    options: Mapping[str, Any] | None = None,
) -> str:
    if think:
        raise ValueError("Thinking is intentionally disabled for this pipeline")
    return _default_client.chat(
        messages,
        output_schema=output_schema,
        options=options,
    )


def installed_models() -> set[str]:
    return _default_client.installed_models()


def unload() -> None:
    _default_client.unload()

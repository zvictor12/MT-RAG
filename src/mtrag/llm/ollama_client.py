from collections.abc import Mapping, Sequence
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from mtrag.config import settings


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
                status_forcelist=(429, 500, 502, 503, 504),
            )
        ),
    )
    return session


class OllamaClient:
    def __init__(
        self,
        *,
        url: str = settings.ollama_url,
        model: str = settings.ollama_model,
        num_ctx: int = settings.ollama_num_ctx,
        num_predict: int = settings.ollama_num_predict,
        seed: int = settings.ollama_seed,
        keep_alive: str = settings.ollama_keep_alive,
        timeout: int = settings.ollama_timeout,
    ) -> None:
        self.url = url
        self.model = model
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.seed = seed
        self.keep_alive = keep_alive
        self.timeout = timeout
        self.session = _session()

    @property
    def model_name(self) -> str:
        return self.model

    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        output_schema: Mapping[str, Any] | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> str:
        request_options: dict[str, Any] = {
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
            "temperature": 0,
            "seed": self.seed,
        }
        if options is not None:
            request_options.update(options)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "stream": False,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": request_options,
        }
        if output_schema is not None:
            payload["format"] = output_schema

        response = self.session.post(
            f"{self.url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def installed_model_digests(self) -> dict[str, str]:
        response = self.session.get(
            f"{self.url}/api/tags",
            timeout=10,
        )
        response.raise_for_status()
        return {
            str(model["name"]): str(model["digest"])
            for model in response.json()["models"]
        }

    def unload(self) -> None:
        response = self.session.post(
            f"{self.url}/api/generate",
            json={
                "model": self.model,
                "prompt": "",
                "stream": False,
                "keep_alive": 0,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

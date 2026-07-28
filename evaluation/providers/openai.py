from __future__ import annotations

import base64
import io
import threading
from typing import Callable, Sequence

from PIL import Image

from evaluation.providers.base import (
    chat_completion_text,
    normalize_provider_answer,
    require_env,
    retry_call,
)


def _jpeg_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class AzureOpenAIProvider:
    provider_name = "azure-openai"

    def __init__(
        self,
        model_id: str | None = None,
        client=None,
        attempts: int = 3,
        base_delay: float = 1.0,
        sleep: Callable[[float], None] | None = None,
        random_source: Callable[[], float] | None = None,
        jitter: float = 0.1,
    ):
        names = [
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_VERSION",
        ]
        if model_id is None:
            names.append("AZURE_OPENAI_DEPLOYMENT")
        config = require_env(names)
        self.model_id = model_id or config["AZURE_OPENAI_DEPLOYMENT"]
        self._api_key = config["AZURE_OPENAI_API_KEY"]
        self._endpoint = config["AZURE_OPENAI_ENDPOINT"]
        self._api_version = config["AZURE_OPENAI_API_VERSION"]
        self._client = client
        self._attempts = attempts
        self._base_delay = base_delay
        self._sleep = sleep
        self._random_source = random_source
        self._jitter = jitter
        self._transient_exceptions = ()

    def _get_client(self):
        if self._client is None:
            import openai

            self._transient_exceptions = tuple(
                error_type
                for name in ("APIConnectionError", "APITimeoutError")
                if isinstance(
                    (error_type := getattr(openai, name, None)),
                    type,
                )
            )
            self._client = openai.AzureOpenAI(
                api_key=self._api_key,
                azure_endpoint=self._endpoint,
                api_version=self._api_version,
                max_retries=0,
            )
        return self._client

    def generate(
        self,
        prompt: str,
        frames: Sequence[Image.Image],
    ) -> dict:
        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64," + _jpeg_base64(frame)
                },
            }
            for frame in frames
        ]
        content.append({"type": "text", "text": prompt})
        client = self._get_client()

        def request():
            return client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": content}],
                max_tokens=2048,
                temperature=0.9,
                top_p=0.9,
            )

        response = retry_call(
            request,
            attempts=self._attempts,
            base_delay=self._base_delay,
            sleep=self._sleep,
            random_source=self._random_source,
            jitter=self._jitter,
            provider=self.provider_name,
            transient_exceptions=self._transient_exceptions,
        )
        text = chat_completion_text(response, self.provider_name)
        return normalize_provider_answer(text, self.provider_name)


class OpenAIJudgeProvider:
    provider_name = "openai"

    def __init__(
        self,
        model_id: str | None = None,
        client=None,
        attempts: int = 3,
        base_delay: float = 1.0,
        sleep: Callable[[float], None] | None = None,
        random_source: Callable[[], float] | None = None,
        jitter: float = 0.1,
    ):
        names = ["OPENAI_API_KEY"]
        if model_id is None:
            names.append("OPENAI_MODEL")
        config = require_env(names)
        self.model_id = model_id or config["OPENAI_MODEL"]
        self._api_key = config["OPENAI_API_KEY"]
        self._client = client
        self._attempts = attempts
        self._base_delay = base_delay
        self._sleep = sleep
        self._random_source = random_source
        self._jitter = jitter
        self._transient_exceptions = ()
        self._client_lock = threading.Lock()

    def _get_client(self):
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    import openai

                    transient_exceptions = tuple(
                        error_type
                        for name in (
                            "APIConnectionError",
                            "APITimeoutError",
                        )
                        if isinstance(
                            (error_type := getattr(openai, name, None)),
                            type,
                        )
                    )
                    client = openai.OpenAI(
                        api_key=self._api_key,
                        max_retries=0,
                    )
                    self._transient_exceptions = transient_exceptions
                    self._client = client
        return self._client

    def complete(
        self,
        system_prompt: str,
        prompt: str,
        json_output: bool,
    ) -> str:
        request_options = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        if json_output:
            request_options["response_format"] = {"type": "json_object"}
        client = self._get_client()

        response = retry_call(
            lambda: client.chat.completions.create(
                **request_options
            ),
            attempts=self._attempts,
            base_delay=self._base_delay,
            sleep=self._sleep,
            random_source=self._random_source,
            jitter=self._jitter,
            provider=self.provider_name,
            transient_exceptions=self._transient_exceptions,
        )
        return chat_completion_text(response, self.provider_name)

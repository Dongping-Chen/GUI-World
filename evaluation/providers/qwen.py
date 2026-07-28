from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Mapping, Sequence

from PIL import Image

from evaluation.providers.base import (
    ProviderRequestError,
    ProviderResponseError,
    normalize_provider_answer,
    require_env,
    require_response_text,
    retry_call,
)


def _value(container, key):
    if isinstance(container, Mapping):
        return container[key]
    return getattr(container, key)


class QwenProvider:
    provider_name = "qwen"

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
        names = ["DASHSCOPE_API_KEY"]
        if model_id is None:
            names.append("DASHSCOPE_MODEL")
        config = require_env(names)
        self.model_id = model_id or config["DASHSCOPE_MODEL"]
        self._api_key = config["DASHSCOPE_API_KEY"]
        self._base_url = os.environ.get("DASHSCOPE_BASE_URL")
        self._client = client
        self._attempts = attempts
        self._base_delay = base_delay
        self._sleep = sleep
        self._random_source = random_source
        self._jitter = jitter
        self._transient_exceptions = ()

    def _get_client(self):
        if self._client is None:
            import dashscope
            import requests

            self._transient_exceptions = (
                requests.ConnectionError,
                requests.Timeout,
            )
            self._client = dashscope
        return self._client

    def generate(
        self,
        prompt: str,
        frames: Sequence[Image.Image],
    ) -> dict:
        with TemporaryDirectory(prefix="gui-world-qwen-") as directory:
            root = Path(directory)
            image_uris = []
            for index, frame in enumerate(frames):
                path = root / f"{index}.jpg"
                frame.convert("RGB").save(path, format="JPEG")
                image_uris.append(path.resolve().as_uri())
            content = [{"image": uri} for uri in image_uris]
            content.append({"text": prompt})
            messages = [{"role": "user", "content": content}]
            client = self._get_client()

            def request():
                conversation = getattr(
                    client,
                    "MultiModalConversation",
                    client,
                )
                request_options = {
                    "model": self.model_id,
                    "messages": messages,
                    "api_key": self._api_key,
                }
                if self._base_url:
                    request_options["base_address"] = self._base_url
                response = conversation.call(**request_options)
                if isinstance(response, Mapping):
                    status_code = response.get("status_code", 200)
                    code = response.get("code", status_code)
                    message = response.get("message", "unknown error")
                else:
                    status_code = getattr(response, "status_code", 200)
                    code = getattr(response, "code", status_code)
                    message = getattr(response, "message", "unknown error")
                if status_code != 200:
                    raise ProviderRequestError(
                        self.provider_name,
                        f"{code}: {message}",
                        status_code=status_code,
                    )
                return response

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
            try:
                output = _value(response, "output")
                choices = _value(output, "choices")
                if (
                    not isinstance(choices, Sequence)
                    or isinstance(choices, (str, bytes))
                    or not choices
                ):
                    raise ProviderResponseError(
                        self.provider_name,
                        "choices must contain at least one completion.",
                    )
                message = _value(choices[0], "message")
                response_content = _value(message, "content")
                if (
                    not isinstance(response_content, Sequence)
                    or isinstance(response_content, (str, bytes))
                    or not response_content
                ):
                    raise ProviderResponseError(
                        self.provider_name,
                        "message content must contain at least one block.",
                    )
                text = next(
                    (
                        _value(block, "text")
                        for block in response_content
                        if (
                            isinstance(block, Mapping)
                            and "text" in block
                        )
                        or hasattr(block, "text")
                    ),
                    None,
                )
            except ProviderResponseError:
                raise
            except (AttributeError, KeyError, TypeError) as error:
                raise ProviderResponseError(
                    self.provider_name,
                    "response is missing choices or message content.",
                ) from error
            text = require_response_text(
                text,
                self.provider_name,
                "message content text",
            )
            return normalize_provider_answer(text, self.provider_name)

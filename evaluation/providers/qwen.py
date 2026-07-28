from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Mapping, Sequence

from PIL import Image

from evaluation.parsing import normalize_answer
from evaluation.providers.base import require_env, retry_call


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

    def _get_client(self):
        if self._client is None:
            import dashscope

            if self._base_url:
                dashscope.base_http_api_url = self._base_url
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

            def request():
                client = self._get_client()
                conversation = getattr(
                    client,
                    "MultiModalConversation",
                    client,
                )
                response = conversation.call(
                    model=self.model_id,
                    messages=messages,
                    api_key=self._api_key,
                )
                if isinstance(response, Mapping):
                    status_code = response.get("status_code", 200)
                    code = response.get("code", status_code)
                    message = response.get("message", "unknown error")
                else:
                    status_code = getattr(response, "status_code", 200)
                    code = getattr(response, "code", status_code)
                    message = getattr(response, "message", "unknown error")
                if status_code != 200:
                    raise RuntimeError(
                        f"DashScope request failed ({code}): {message}"
                    )
                return response

            response = retry_call(
                request,
                attempts=self._attempts,
                base_delay=self._base_delay,
                sleep=self._sleep,
            )
            output = _value(response, "output")
            choice = _value(output, "choices")[0]
            message = _value(choice, "message")
            response_content = _value(message, "content")
            text = _value(response_content[0], "text")
            return normalize_answer(text)

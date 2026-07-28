from __future__ import annotations

import base64
import io
from typing import Callable, Sequence

from PIL import Image

from evaluation.parsing import normalize_answer
from evaluation.providers.base import require_env, retry_call


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

    def _get_client(self):
        if self._client is None:
            from openai import AzureOpenAI

            self._client = AzureOpenAI(
                api_key=self._api_key,
                azure_endpoint=self._endpoint,
                api_version=self._api_version,
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

        def request():
            return self._get_client().chat.completions.create(
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
        )
        return normalize_answer(response.choices[0].message.content)


class OpenAIJudgeProvider:
    provider_name = "openai"

    def __init__(
        self,
        model_id: str | None = None,
        client=None,
        attempts: int = 3,
        base_delay: float = 1.0,
        sleep: Callable[[float], None] | None = None,
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

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key)
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

        response = retry_call(
            lambda: self._get_client().chat.completions.create(
                **request_options
            ),
            attempts=self._attempts,
            base_delay=self._base_delay,
            sleep=self._sleep,
        )
        return response.choices[0].message.content

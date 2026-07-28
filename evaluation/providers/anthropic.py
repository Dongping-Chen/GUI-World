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


class AnthropicProvider:
    provider_name = "anthropic"

    def __init__(
        self,
        model_id: str | None = None,
        client=None,
        attempts: int = 3,
        base_delay: float = 1.0,
        sleep: Callable[[float], None] | None = None,
    ):
        names = ["ANTHROPIC_API_KEY"]
        if model_id is None:
            names.append("ANTHROPIC_MODEL")
        config = require_env(names)
        self.model_id = model_id or config["ANTHROPIC_MODEL"]
        self._api_key = config["ANTHROPIC_API_KEY"]
        self._client = client
        self._attempts = attempts
        self._base_delay = base_delay
        self._sleep = sleep

    def _get_client(self):
        if self._client is None:
            from anthropic import Anthropic

            self._client = Anthropic(api_key=self._api_key)
        return self._client

    def generate(
        self,
        prompt: str,
        frames: Sequence[Image.Image],
    ) -> dict:
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": _jpeg_base64(frame),
                },
            }
            for frame in frames
        ]
        content.append({"type": "text", "text": prompt})

        response = retry_call(
            lambda: self._get_client().messages.create(
                model=self.model_id,
                max_tokens=2048,
                temperature=0.9,
                messages=[{"role": "user", "content": content}],
            ),
            attempts=self._attempts,
            base_delay=self._base_delay,
            sleep=self._sleep,
        )
        text = next(
            block.text
            for block in response.content
            if getattr(block, "text", None) is not None
        )
        return normalize_answer(text)

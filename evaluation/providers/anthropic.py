from __future__ import annotations

import base64
import io
from typing import Callable, Sequence

from PIL import Image

from evaluation.providers.base import (
    ProviderResponseError,
    normalize_provider_answer,
    require_env,
    retry_call,
)


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
        random_source: Callable[[], float] | None = None,
        jitter: float = 0.1,
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
        self._random_source = random_source
        self._jitter = jitter
        self._transient_exceptions = ()

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._transient_exceptions = tuple(
                error_type
                for name in ("APIConnectionError", "APITimeoutError")
                if isinstance(
                    (error_type := getattr(anthropic, name, None)),
                    type,
                )
            )
            self._client = anthropic.Anthropic(
                api_key=self._api_key,
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
        client = self._get_client()

        response = retry_call(
            lambda: client.messages.create(
                model=self.model_id,
                max_tokens=2048,
                temperature=0.9,
                messages=[{"role": "user", "content": content}],
            ),
            attempts=self._attempts,
            base_delay=self._base_delay,
            sleep=self._sleep,
            random_source=self._random_source,
            jitter=self._jitter,
            provider=self.provider_name,
            transient_exceptions=self._transient_exceptions,
        )
        blocks = getattr(response, "content", None)
        if not isinstance(blocks, Sequence) or isinstance(
            blocks,
            (str, bytes),
        ):
            raise ProviderResponseError(
                self.provider_name,
                "content must be a sequence of response blocks.",
            )
        text = None
        for block in blocks:
            candidate = (
                block.get("text")
                if isinstance(block, dict)
                else getattr(block, "text", None)
            )
            if isinstance(candidate, str) and candidate.strip():
                text = candidate
                break
        if text is None:
            raise ProviderResponseError(
                self.provider_name,
                "content does not contain a non-empty text block.",
            )
        return normalize_provider_answer(text, self.provider_name)

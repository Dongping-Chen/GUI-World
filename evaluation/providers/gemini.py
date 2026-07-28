from __future__ import annotations

from typing import Callable, Sequence

from PIL import Image

from evaluation.providers.base import (
    ProviderResponseError,
    normalize_provider_answer,
    require_env,
    require_response_text,
    retry_call,
)


class GeminiProvider:
    provider_name = "gemini"

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
        names = ["GEMINI_API_KEY"]
        if model_id is None:
            names.append("GEMINI_MODEL")
        config = require_env(names)
        self.model_id = model_id or config["GEMINI_MODEL"]
        self._api_key = config["GEMINI_API_KEY"]
        self._client = client
        self._attempts = attempts
        self._base_delay = base_delay
        self._sleep = sleep
        self._random_source = random_source
        self._jitter = jitter
        self._transient_exceptions = ()

    def _get_client(self):
        if self._client is None:
            from google import genai
            import httpx
            import requests

            self._transient_exceptions = (
                httpx.NetworkError,
                httpx.TimeoutException,
                requests.ConnectionError,
                requests.Timeout,
            )
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def generate(
        self,
        prompt: str,
        frames: Sequence[Image.Image],
    ) -> dict:
        contents = [*frames, prompt]
        client = self._get_client()
        response = retry_call(
            lambda: client.models.generate_content(
                model=self.model_id,
                contents=contents,
            ),
            attempts=self._attempts,
            base_delay=self._base_delay,
            sleep=self._sleep,
            random_source=self._random_source,
            jitter=self._jitter,
            provider=self.provider_name,
            transient_exceptions=self._transient_exceptions,
        )
        try:
            text = response.text
        except Exception as error:
            raise ProviderResponseError(
                self.provider_name,
                f"text is unavailable: {error}",
            ) from error
        if not isinstance(text, str) or not text.strip():
            feedback = getattr(response, "prompt_feedback", None)
            block_reason = getattr(feedback, "block_reason", None)
            if block_reason:
                raise ProviderResponseError(
                    self.provider_name,
                    f"response was blocked: {block_reason}",
                )
        text = require_response_text(
            text,
            self.provider_name,
            "generated text",
        )
        return normalize_provider_answer(text, self.provider_name)

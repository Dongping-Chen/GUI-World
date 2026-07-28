from __future__ import annotations

from typing import Callable, Sequence

from PIL import Image

from evaluation.parsing import normalize_answer
from evaluation.providers.base import require_env, retry_call


class GeminiProvider:
    provider_name = "gemini"

    def __init__(
        self,
        model_id: str | None = None,
        client=None,
        attempts: int = 3,
        base_delay: float = 1.0,
        sleep: Callable[[float], None] | None = None,
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

    def _get_client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def generate(
        self,
        prompt: str,
        frames: Sequence[Image.Image],
    ) -> dict:
        contents = [*frames, prompt]
        response = retry_call(
            lambda: self._get_client().models.generate_content(
                model=self.model_id,
                contents=contents,
            ),
            attempts=self._attempts,
            base_delay=self._base_delay,
            sleep=self._sleep,
        )
        return normalize_answer(response.text)

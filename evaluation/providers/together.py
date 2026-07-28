from __future__ import annotations

from typing import Callable

from evaluation.providers.base import require_env, retry_call


class TogetherJudgeProvider:
    provider_name = "together"

    def __init__(
        self,
        model_id: str | None = None,
        client=None,
        attempts: int = 3,
        base_delay: float = 1.0,
        sleep: Callable[[float], None] | None = None,
    ):
        names = ["TOGETHER_API_KEY"]
        if model_id is None:
            names.append("TOGETHER_MODEL")
        config = require_env(names)
        self.model_id = model_id or config["TOGETHER_MODEL"]
        self._api_key = config["TOGETHER_API_KEY"]
        self._client = client
        self._attempts = attempts
        self._base_delay = base_delay
        self._sleep = sleep

    def _get_client(self):
        if self._client is None:
            from together import Together

            self._client = Together(api_key=self._api_key)
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

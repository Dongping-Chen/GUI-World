from __future__ import annotations

import os
import time
from typing import Callable, Protocol, Sequence, TypeVar

from PIL import Image


class ProviderConfigurationError(RuntimeError):
    """Raised when a selected provider is missing required configuration."""


class InferenceProvider(Protocol):
    def generate(
        self,
        prompt: str,
        frames: Sequence[Image.Image],
    ) -> dict:
        """Generate and normalize an answer from a sequence of GUI frames."""


class JudgeProvider(Protocol):
    def complete(
        self,
        system_prompt: str,
        prompt: str,
        json_output: bool,
    ) -> str:
        """Return the raw judge response text."""


def require_env(names: Sequence[str]) -> dict[str, str]:
    """Return required environment values or report every missing name."""

    values = {}
    missing = []
    for name in names:
        value = os.environ.get(name)
        if value is None or not value.strip():
            missing.append(name)
        else:
            values[name] = value
    if missing:
        raise ProviderConfigurationError(
            "Missing required environment variables: " + ", ".join(missing)
        )
    return values


_Result = TypeVar("_Result")


def retry_call(
    function: Callable[[], _Result],
    attempts: int = 3,
    base_delay: float = 1.0,
    sleep: Callable[[float], None] | None = None,
) -> _Result:
    """Call *function* with bounded exponential-backoff retries."""

    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise ValueError("Retry attempts must be a positive integer.")
    if base_delay < 0:
        raise ValueError("Retry base delay must be non-negative.")

    sleep_function = time.sleep if sleep is None else sleep
    for attempt in range(attempts):
        try:
            return function()
        except Exception:
            if attempt == attempts - 1:
                raise
            sleep_function(base_delay * (2**attempt))
    raise RuntimeError("Retry loop ended unexpectedly.")

from __future__ import annotations

import math
import os
import random
import time
from typing import Callable, Mapping, Protocol, Sequence, TypeVar

from PIL import Image

from evaluation.parsing import ResponseParseError, normalize_answer


class ProviderError(RuntimeError):
    """Base class for controlled provider failures."""


class ProviderConfigurationError(ProviderError):
    """Raised when a selected provider is missing required configuration."""


class ProviderRequestError(ProviderError):
    """Raised when a provider request fails before a valid response."""

    def __init__(
        self,
        provider: str,
        message: str,
        *,
        status_code: int | None = None,
        retry_after: float | str | None = None,
        transient: bool | None = None,
    ):
        self.provider = provider
        self.status_code = status_code
        self.retry_after = _numeric_retry_after(retry_after)
        self.transient = (
            _is_transient_status(status_code)
            if transient is None
            else transient
        )
        super().__init__(f"{provider} request failed: {message}")


class ProviderResponseError(ProviderError):
    """Raised when a provider returns a missing or malformed response."""

    def __init__(self, provider: str, message: str):
        self.provider = provider
        super().__init__(f"{provider} response invalid: {message}")


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
        """Return raw judge text.

        Implementations, including injected providers, must be thread-safe.
        """


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


def chat_completion_text(response, provider: str) -> str:
    """Extract non-empty text from a Chat Completions response."""

    choices = _response_field(response, "choices", provider)
    if (
        not isinstance(choices, Sequence)
        or isinstance(choices, (str, bytes))
        or not choices
    ):
        raise ProviderResponseError(
            provider,
            "choices must contain at least one completion.",
        )
    message = _response_field(choices[0], "message", provider)
    refusal = _optional_response_field(message, "refusal")
    if isinstance(refusal, str) and refusal.strip():
        raise ProviderResponseError(
            provider,
            f"request was refused: {refusal.strip()}",
        )
    content = _response_field(message, "content", provider)
    return require_response_text(content, provider, "message content")


def require_response_text(value, provider: str, field: str) -> str:
    """Return non-empty response text or raise a controlled response error."""

    if not isinstance(value, str) or not value.strip():
        raise ProviderResponseError(
            provider,
            f"{field} must be a non-empty string.",
        )
    return value


def normalize_provider_answer(value, provider: str) -> dict[str, str]:
    """Normalize inference output under the provider error hierarchy."""

    try:
        return normalize_answer(value)
    except ResponseParseError as error:
        raise ProviderResponseError(provider, str(error)) from error


_Result = TypeVar("_Result")


def retry_call(
    function: Callable[[], _Result],
    attempts: int = 3,
    base_delay: float = 1.0,
    sleep: Callable[[float], None] | None = None,
    random_source: Callable[[], float] | None = None,
    jitter: float = 0.1,
    max_delay: float = 60.0,
    provider: str | None = None,
    transient_exceptions: tuple[type[BaseException], ...] = (),
) -> _Result:
    """Call *function* with bounded exponential-backoff retries."""

    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise ValueError("Retry attempts must be a positive integer.")
    if not math.isfinite(base_delay):
        raise ValueError("Retry base delay must be finite.")
    if base_delay < 0:
        raise ValueError("Retry base delay must be non-negative.")
    if not math.isfinite(jitter):
        raise ValueError("Retry jitter must be finite.")
    if jitter < 0 or jitter > 1:
        raise ValueError("Retry jitter must be between zero and one.")
    if not math.isfinite(max_delay):
        raise ValueError("Retry maximum delay must be finite.")
    if max_delay < 0:
        raise ValueError("Retry maximum delay must be non-negative.")

    sleep_function = time.sleep if sleep is None else sleep
    random_function = random.random if random_source is None else random_source
    for attempt in range(attempts):
        try:
            return function()
        except Exception as error:
            transient = _is_transient_error(error, transient_exceptions)
            controlled = _controlled_request_error(error, provider, transient)
            if not transient or attempt == attempts - 1:
                if controlled is error:
                    raise
                raise controlled from error
            retry_after = _retry_after_from_error(error)
            backoff = min(max_delay, base_delay * (2**attempt))
            delay = max(backoff, retry_after or 0.0)
            random_value = min(1.0, max(0.0, float(random_function())))
            delay += delay * jitter * random_value
            sleep_function(min(max_delay, delay))
    raise RuntimeError("Retry loop ended unexpectedly.")


def _controlled_request_error(
    error: Exception,
    provider: str | None,
    transient: bool,
) -> Exception:
    if isinstance(error, ProviderError) or provider is None:
        return error
    status_code = _status_code(error)
    message = str(error).strip() or type(error).__name__
    return ProviderRequestError(
        provider,
        message,
        status_code=status_code,
        retry_after=_retry_after_from_error(error),
        transient=transient,
    )


def _is_transient_error(
    error: Exception,
    transient_exceptions: tuple[type[BaseException], ...],
) -> bool:
    if isinstance(error, ProviderResponseError):
        return False
    if isinstance(error, ProviderRequestError):
        return error.transient
    status_code = _status_code(error)
    if status_code is not None:
        return _is_transient_status(status_code)
    connection_types = (TimeoutError, ConnectionError) + transient_exceptions
    return isinstance(error, connection_types)


def _is_transient_status(status_code: int | None) -> bool:
    if status_code is None:
        return False
    return status_code in {408, 409, 429} or 500 <= status_code <= 599


def _status_code(error: Exception) -> int | None:
    for name in ("status_code", "code"):
        value = getattr(error, name, None)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def _retry_after_from_error(error: Exception) -> float | None:
    if isinstance(error, ProviderRequestError):
        return error.retry_after
    direct = _numeric_retry_after(getattr(error, "retry_after", None))
    if direct is not None:
        return direct
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        headers = getattr(error, "headers", None)
    if not isinstance(headers, Mapping):
        return None
    for name, value in headers.items():
        if str(name).casefold() == "retry-after":
            return _numeric_retry_after(value)
    return None


def _numeric_retry_after(value) -> float | None:
    try:
        retry_after = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(retry_after) or retry_after < 0:
        return None
    return retry_after


def _response_field(container, key: str, provider: str):
    if isinstance(container, Mapping):
        if key in container:
            return container[key]
    else:
        try:
            return getattr(container, key)
        except (AttributeError, TypeError):
            pass
    raise ProviderResponseError(provider, f"response is missing {key}.")


def _optional_response_field(container, key: str):
    if isinstance(container, Mapping):
        return container.get(key)
    return getattr(container, key, None)

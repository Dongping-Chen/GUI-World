from __future__ import annotations

import importlib

from evaluation.providers.base import (
    InferenceProvider,
    JudgeProvider,
    ProviderConfigurationError,
    ProviderError,
    ProviderRequestError,
    ProviderResponseError,
)


_ALIASES = {
    "azure-openai": "azure-openai",
    "gpt-4v": "azure-openai",
    "anthropic": "anthropic",
    "claude3-opus": "anthropic",
    "gemini": "gemini",
    "gemini-pro": "gemini",
    "qwen": "qwen",
    "qwen-vl-max": "qwen",
    "openai": "openai",
    "gpt-4": "openai",
    "together": "together",
    "llama3": "together",
}

_INFERENCE_ADAPTERS = {
    "azure-openai": (
        "evaluation.providers.openai",
        "AzureOpenAIProvider",
    ),
    "anthropic": (
        "evaluation.providers.anthropic",
        "AnthropicProvider",
    ),
    "gemini": (
        "evaluation.providers.gemini",
        "GeminiProvider",
    ),
    "qwen": (
        "evaluation.providers.qwen",
        "QwenProvider",
    ),
}

_JUDGE_ADAPTERS = {
    "openai": (
        "evaluation.providers.openai",
        "OpenAIJudgeProvider",
    ),
    "together": (
        "evaluation.providers.together",
        "TogetherJudgeProvider",
    ),
}


def _canonical_provider(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError(f"Unknown provider: {name!r}")
    normalized = name.strip().casefold()
    try:
        return _ALIASES[normalized]
    except KeyError as error:
        raise ValueError(f"Unknown provider: {name!r}") from error


def _create_provider(name, operation, adapters, **options):
    canonical = _canonical_provider(name)
    if canonical not in adapters:
        raise ValueError(
            f"Provider {name!r} does not support {operation}."
        )
    module_name, class_name = adapters[canonical]
    module = importlib.import_module(module_name)
    provider_class = getattr(module, class_name)
    return provider_class(**options)


def create_inference_provider(
    name: str,
    model_id: str | None = None,
    client=None,
    attempts: int = 3,
    base_delay: float = 1.0,
    sleep=None,
    random_source=None,
    jitter: float = 0.1,
) -> InferenceProvider:
    """Create an inference adapter from a canonical or legacy name."""

    return _create_provider(
        name,
        "inference",
        _INFERENCE_ADAPTERS,
        model_id=model_id,
        client=client,
        attempts=attempts,
        base_delay=base_delay,
        sleep=sleep,
        random_source=random_source,
        jitter=jitter,
    )


def create_judge_provider(
    name: str,
    model_id: str | None = None,
    client=None,
    attempts: int = 3,
    base_delay: float = 1.0,
    sleep=None,
    random_source=None,
    jitter: float = 0.1,
) -> JudgeProvider:
    """Create a judge adapter from a canonical or legacy name."""

    return _create_provider(
        name,
        "judge",
        _JUDGE_ADAPTERS,
        model_id=model_id,
        client=client,
        attempts=attempts,
        base_delay=base_delay,
        sleep=sleep,
        random_source=random_source,
        jitter=jitter,
    )


__all__ = [
    "InferenceProvider",
    "JudgeProvider",
    "ProviderConfigurationError",
    "ProviderError",
    "ProviderRequestError",
    "ProviderResponseError",
    "create_inference_provider",
    "create_judge_provider",
]

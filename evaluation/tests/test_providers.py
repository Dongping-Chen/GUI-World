from __future__ import annotations

import base64
import importlib
import io
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image


ALL_PROVIDER_ENV = {
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_DEPLOYMENT",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL",
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "DASHSCOPE_API_KEY",
    "DASHSCOPE_MODEL",
    "DASHSCOPE_BASE_URL",
    "TOGETHER_API_KEY",
    "TOGETHER_MODEL",
}

REQUIRED_ENV_BY_PROVIDER = {
    ("inference", "azure-openai"): (
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_DEPLOYMENT",
    ),
    ("inference", "anthropic"): (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_MODEL",
    ),
    ("inference", "gemini"): (
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
    ),
    ("inference", "qwen"): (
        "DASHSCOPE_API_KEY",
        "DASHSCOPE_MODEL",
    ),
    ("judge", "openai"): (
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
    ),
    ("judge", "together"): (
        "TOGETHER_API_KEY",
        "TOGETHER_MODEL",
    ),
}


def _clear_provider_env(monkeypatch):
    for name in ALL_PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)


def _set_env(monkeypatch, names):
    for name in names:
        monkeypatch.setenv(name, f"test-{name.casefold()}")


def _chat_client(response_text):
    recorder = SimpleNamespace(calls=[])

    def create(**kwargs):
        recorder.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=response_text)
                )
            ]
        )

    recorder.create = create
    return SimpleNamespace(
        chat=SimpleNamespace(completions=recorder),
        recorder=recorder,
    )


def _frame(color="red"):
    return Image.new("RGB", (3, 2), color=color)


def test_require_env_reports_every_missing_name(monkeypatch):
    from evaluation.providers.base import (
        ProviderConfigurationError,
        require_env,
    )

    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("PRESENT", "configured")

    with pytest.raises(ProviderConfigurationError) as error:
        require_env(("MISSING_ONE", "PRESENT", "MISSING_TWO"))

    message = str(error.value)
    assert "MISSING_ONE" in message
    assert "MISSING_TWO" in message
    assert "PRESENT" not in message


@pytest.mark.parametrize(
    ("operation", "name", "required_names"),
    [
        (operation, name, required_names)
        for (operation, name), required_names
        in REQUIRED_ENV_BY_PROVIDER.items()
    ],
)
def test_factories_report_all_missing_provider_configuration(
    monkeypatch,
    operation,
    name,
    required_names,
):
    from evaluation.providers import (
        create_inference_provider,
        create_judge_provider,
    )
    from evaluation.providers.base import ProviderConfigurationError

    _clear_provider_env(monkeypatch)
    factory = (
        create_inference_provider
        if operation == "inference"
        else create_judge_provider
    )

    with pytest.raises(ProviderConfigurationError) as error:
        factory(name)

    for variable in required_names:
        assert variable in str(error.value)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("azure-openai", "azure-openai"),
        ("GPT-4V", "azure-openai"),
        ("anthropic", "anthropic"),
        ("Claude3-Opus", "anthropic"),
        ("gemini", "gemini"),
        ("Gemini-Pro", "gemini"),
        ("qwen", "qwen"),
        ("Qwen-VL-Max", "qwen"),
    ],
)
def test_inference_factory_accepts_canonical_and_legacy_aliases(
    monkeypatch,
    name,
    expected,
):
    from evaluation.providers import create_inference_provider

    _clear_provider_env(monkeypatch)
    canonical_names = REQUIRED_ENV_BY_PROVIDER[("inference", expected)]
    _set_env(monkeypatch, canonical_names)

    provider = create_inference_provider(name, client=object())

    assert provider.provider_name == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("openai", "openai"),
        ("gpt-4", "openai"),
        ("together", "together"),
        ("llama3", "together"),
    ],
)
def test_judge_factory_accepts_canonical_and_legacy_aliases(
    monkeypatch,
    name,
    expected,
):
    from evaluation.providers import create_judge_provider

    _clear_provider_env(monkeypatch)
    canonical_names = REQUIRED_ENV_BY_PROVIDER[("judge", expected)]
    _set_env(monkeypatch, canonical_names)

    provider = create_judge_provider(name, client=object())

    assert provider.provider_name == expected


@pytest.mark.parametrize(
    ("operation", "name"),
    [
        ("inference", "openai"),
        ("inference", "together"),
        ("judge", "azure-openai"),
        ("judge", "anthropic"),
        ("judge", "gemini"),
        ("judge", "qwen"),
    ],
)
def test_factory_rejects_unsupported_operation_before_import(
    monkeypatch,
    operation,
    name,
):
    import evaluation.providers as providers

    imports = []

    def fail_import(module_name):
        imports.append(module_name)
        raise AssertionError("factory imported an adapter before rejecting it")

    monkeypatch.setattr(providers.importlib, "import_module", fail_import)
    factory = (
        providers.create_inference_provider
        if operation == "inference"
        else providers.create_judge_provider
    )

    with pytest.raises(ValueError, match="does not support"):
        factory(name)

    assert imports == []


@pytest.mark.parametrize(
    ("operation", "name", "required_names", "model_variable"),
    [
        (
            "inference",
            "azure-openai",
            REQUIRED_ENV_BY_PROVIDER[("inference", "azure-openai")],
            "AZURE_OPENAI_DEPLOYMENT",
        ),
        (
            "inference",
            "anthropic",
            REQUIRED_ENV_BY_PROVIDER[("inference", "anthropic")],
            "ANTHROPIC_MODEL",
        ),
        (
            "inference",
            "gemini",
            REQUIRED_ENV_BY_PROVIDER[("inference", "gemini")],
            "GEMINI_MODEL",
        ),
        (
            "inference",
            "qwen",
            REQUIRED_ENV_BY_PROVIDER[("inference", "qwen")],
            "DASHSCOPE_MODEL",
        ),
        (
            "judge",
            "openai",
            REQUIRED_ENV_BY_PROVIDER[("judge", "openai")],
            "OPENAI_MODEL",
        ),
        (
            "judge",
            "together",
            REQUIRED_ENV_BY_PROVIDER[("judge", "together")],
            "TOGETHER_MODEL",
        ),
    ],
)
def test_model_override_takes_priority_and_removes_model_env_requirement(
    monkeypatch,
    operation,
    name,
    required_names,
    model_variable,
):
    from evaluation.providers import (
        create_inference_provider,
        create_judge_provider,
    )

    _clear_provider_env(monkeypatch)
    _set_env(
        monkeypatch,
        [variable for variable in required_names if variable != model_variable],
    )
    factory = (
        create_inference_provider
        if operation == "inference"
        else create_judge_provider
    )

    provider = factory(name, model_id="explicit-model", client=object())

    assert provider.model_id == "explicit-model"


def test_azure_openai_uses_jpeg_data_urls_and_normalizes_response(monkeypatch):
    from evaluation.providers import create_inference_provider

    _clear_provider_env(monkeypatch)
    _set_env(
        monkeypatch,
        REQUIRED_ENV_BY_PROVIDER[("inference", "azure-openai")],
    )
    client = _chat_client(
        '```json\n{"Description": "A menu opens.", "Answer": "B"}\n```'
    )
    provider = create_inference_provider(
        "azure-openai",
        model_id="vision-deployment",
        client=client,
    )

    result = provider.generate("Which item opens?", [_frame(), _frame("blue")])

    assert result == {"Description": "A menu opens.", "Answer": "B"}
    request = client.recorder.calls[0]
    assert request["model"] == "vision-deployment"
    content = request["messages"][0]["content"]
    assert [part["type"] for part in content] == [
        "image_url",
        "image_url",
        "text",
    ]
    assert content[-1] == {"type": "text", "text": "Which item opens?"}
    data_url = content[0]["image_url"]["url"]
    assert data_url.startswith("data:image/jpeg;base64,")
    encoded = data_url.partition(",")[2]
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as decoded:
        assert decoded.format == "JPEG"


def test_anthropic_places_jpeg_images_before_text_and_normalizes(monkeypatch):
    from evaluation.providers import create_inference_provider

    _clear_provider_env(monkeypatch)
    _set_env(
        monkeypatch,
        REQUIRED_ENV_BY_PROVIDER[("inference", "anthropic")],
    )
    recorder = SimpleNamespace(calls=[])

    def create(**kwargs):
        recorder.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(text='{"Answer": "Click Save."}')]
        )

    client = SimpleNamespace(
        messages=SimpleNamespace(create=create),
    )
    provider = create_inference_provider("anthropic", client=client)

    assert provider.generate("What happens?", [_frame()]) == {
        "Answer": "Click Save."
    }
    content = recorder.calls[0]["messages"][0]["content"]
    assert [part["type"] for part in content] == ["image", "text"]
    assert content[0]["source"]["media_type"] == "image/jpeg"
    assert content[1] == {"type": "text", "text": "What happens?"}


def test_gemini_sends_images_before_text_and_normalizes(monkeypatch):
    from evaluation.providers import create_inference_provider

    _clear_provider_env(monkeypatch)
    _set_env(
        monkeypatch,
        REQUIRED_ENV_BY_PROVIDER[("inference", "gemini")],
    )
    recorder = SimpleNamespace(calls=[])

    def generate_content(**kwargs):
        recorder.calls.append(kwargs)
        return SimpleNamespace(text='{"Analysis": "Visible.", "Answer": "A"}')

    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=generate_content),
    )
    provider = create_inference_provider("gemini", client=client)
    frames = [_frame(), _frame("green")]

    assert provider.generate("Choose an option.", frames) == {
        "Analysis": "Visible.",
        "Answer": "A",
    }
    request = recorder.calls[0]
    assert request["model"] == "test-gemini_model"
    assert request["contents"][:-1] == frames
    assert request["contents"][-1] == "Choose an option."


def test_qwen_keeps_temporary_images_alive_for_call_then_removes_them(
    monkeypatch,
):
    from evaluation.providers import create_inference_provider

    _clear_provider_env(monkeypatch)
    _set_env(
        monkeypatch,
        REQUIRED_ENV_BY_PROVIDER[("inference", "qwen")],
    )
    observed_paths = []
    recorder = SimpleNamespace(calls=[])

    def call(**kwargs):
        recorder.calls.append(kwargs)
        content = kwargs["messages"][0]["content"]
        image_parts = content[:-1]
        assert all("image" in part for part in image_parts)
        for part in image_parts:
            uri = part["image"]
            assert uri.startswith("file://")
            path = Path(uri.removeprefix("file://"))
            assert path.is_file()
            observed_paths.append(path)
        assert content[-1] == {"text": "What changed?"}
        return {
            "output": {
                "choices": [
                    {
                        "message": {
                            "content": [{"text": '{"Answer": "The menu opened."}'}]
                        }
                    }
                ]
            }
        }

    client = SimpleNamespace(
        MultiModalConversation=SimpleNamespace(call=call),
    )
    provider = create_inference_provider("qwen", client=client)

    assert provider.generate("What changed?", [_frame(), _frame("yellow")]) == {
        "Answer": "The menu opened."
    }
    assert recorder.calls[0]["api_key"] == "test-dashscope_api_key"
    assert all(not path.exists() for path in observed_paths)


def test_qwen_retries_non_success_dashscope_responses(monkeypatch):
    from evaluation.providers import create_inference_provider

    _clear_provider_env(monkeypatch)
    _set_env(
        monkeypatch,
        REQUIRED_ENV_BY_PROVIDER[("inference", "qwen")],
    )
    responses = [
        SimpleNamespace(
            status_code=429,
            code="Throttled",
            message="try again",
        ),
        {
            "output": {
                "choices": [
                    {
                        "message": {
                            "content": [{"text": '{"Answer": "Recovered."}'}]
                        }
                    }
                ]
            }
        },
    ]
    calls = []

    def call(**kwargs):
        calls.append(kwargs)
        return responses.pop(0)

    client = SimpleNamespace(
        MultiModalConversation=SimpleNamespace(call=call),
    )
    provider = create_inference_provider(
        "qwen",
        client=client,
        attempts=2,
        base_delay=0,
        sleep=lambda _: None,
    )

    assert provider.generate("Retry this.", [_frame()]) == {
        "Answer": "Recovered."
    }
    assert len(calls) == 2


def test_qwen_base_url_is_isolated_per_provider_call(monkeypatch):
    from evaluation.providers import create_inference_provider

    _clear_provider_env(monkeypatch)
    _set_env(
        monkeypatch,
        REQUIRED_ENV_BY_PROVIDER[("inference", "qwen")],
    )
    calls = []

    def call(**kwargs):
        calls.append(kwargs)
        return {
            "output": {
                "choices": [
                    {
                        "message": {
                            "content": [{"text": '{"Answer": "Isolated."}'}]
                        }
                    }
                ]
            }
        }

    fake_dashscope = SimpleNamespace(
        base_http_api_url="https://global.example",
        MultiModalConversation=SimpleNamespace(call=call),
    )
    monkeypatch.setitem(sys.modules, "dashscope", fake_dashscope)

    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://a.example")
    provider_a = create_inference_provider("qwen")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://b.example")
    provider_b = create_inference_provider("qwen")
    monkeypatch.delenv("DASHSCOPE_BASE_URL")
    provider_default = create_inference_provider("qwen")

    provider_a.generate("A", [_frame()])
    provider_b.generate("B", [_frame()])
    provider_a.generate("A again", [_frame()])
    provider_default.generate("Default", [_frame()])

    assert [call.get("base_address") for call in calls] == [
        "https://a.example",
        "https://b.example",
        "https://a.example",
        None,
    ]
    assert "base_address" not in calls[-1]
    assert fake_dashscope.base_http_api_url == "https://global.example"


def test_qwen_custom_endpoint_covers_preprocessing_and_restores(monkeypatch):
    from evaluation.providers import create_inference_provider
    from evaluation.providers.base import ProviderRequestError

    _clear_provider_env(monkeypatch)
    _set_env(
        monkeypatch,
        REQUIRED_ENV_BY_PROVIDER[("inference", "qwen")],
    )
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://custom.example")
    preprocessing_endpoints = []

    def call(**kwargs):
        preprocessing_endpoints.append(fake_dashscope.base_http_api_url)
        assert kwargs["base_address"] == "https://custom.example"
        raise ValueError("simulated preprocessing failure")

    fake_dashscope = SimpleNamespace(
        base_http_api_url="https://global.example",
        MultiModalConversation=SimpleNamespace(call=call),
    )
    monkeypatch.setitem(sys.modules, "dashscope", fake_dashscope)
    provider = create_inference_provider("qwen", attempts=1)

    with pytest.raises(
        ProviderRequestError,
        match="simulated preprocessing failure",
    ):
        provider.generate("Question", [_frame()])

    assert preprocessing_endpoints == ["https://custom.example"]
    assert fake_dashscope.base_http_api_url == "https://global.example"


def test_qwen_serializes_two_custom_endpoints(monkeypatch):
    from evaluation.providers import create_inference_provider

    _clear_provider_env(monkeypatch)
    _set_env(
        monkeypatch,
        REQUIRED_ENV_BY_PROVIDER[("inference", "qwen")],
    )
    first_entered = threading.Event()
    second_entered = threading.Event()
    release_first = threading.Event()
    observations = []

    def call(**kwargs):
        prompt = kwargs["messages"][0]["content"][-1]["text"]
        observations.append(
            (
                prompt,
                fake_dashscope.base_http_api_url,
                kwargs["base_address"],
            )
        )
        if prompt == "A":
            first_entered.set()
            assert release_first.wait(timeout=1)
        else:
            second_entered.set()
        return {
            "output": {
                "choices": [
                    {
                        "message": {
                            "content": [{"text": '{"Answer": "Done."}'}]
                        }
                    }
                ]
            }
        }

    fake_dashscope = SimpleNamespace(
        base_http_api_url="https://global.example",
        MultiModalConversation=SimpleNamespace(call=call),
    )
    monkeypatch.setitem(sys.modules, "dashscope", fake_dashscope)
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://a.example")
    provider_a = create_inference_provider("qwen")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://b.example")
    provider_b = create_inference_provider("qwen")
    errors = []

    def generate(provider, prompt):
        try:
            provider.generate(prompt, [_frame()])
        except Exception as error:
            errors.append(error)

    thread_a = threading.Thread(target=generate, args=(provider_a, "A"))
    thread_b = threading.Thread(target=generate, args=(provider_b, "B"))
    thread_a.start()
    assert first_entered.wait(timeout=1)
    thread_b.start()
    second_was_serialized = not second_entered.wait(timeout=0.05)
    release_first.set()
    thread_a.join(timeout=1)
    thread_b.join(timeout=1)

    assert second_was_serialized
    assert not errors
    assert observations == [
        ("A", "https://a.example", "https://a.example"),
        ("B", "https://b.example", "https://b.example"),
    ]
    assert fake_dashscope.base_http_api_url == "https://global.example"


@pytest.mark.parametrize("provider_name", ["openai", "together"])
def test_judge_adapters_return_raw_text_and_request_json_mode(
    monkeypatch,
    provider_name,
):
    from evaluation.providers import create_judge_provider

    _clear_provider_env(monkeypatch)
    _set_env(
        monkeypatch,
        REQUIRED_ENV_BY_PROVIDER[("judge", provider_name)],
    )
    client = _chat_client('{"Evaluation": "Equivalent.", "Score": 5}')
    provider = create_judge_provider(provider_name, client=client)

    result = provider.complete(
        "Evaluate the answer.",
        "Gold: B; predicted: B",
        json_output=True,
    )

    assert result == '{"Evaluation": "Equivalent.", "Score": 5}'
    request = client.recorder.calls[0]
    assert request["messages"] == [
        {"role": "system", "content": "Evaluate the answer."},
        {"role": "user", "content": "Gold: B; predicted: B"},
    ]
    assert request["response_format"] == {"type": "json_object"}


def test_retry_call_uses_bounded_exponential_backoff():
    from evaluation.providers.base import ProviderRequestError, retry_call

    calls = []
    delays = []

    def operation():
        calls.append("called")
        if len(calls) < 3:
            raise ProviderRequestError(
                "test-provider",
                f"failure-{len(calls)}",
                status_code=503,
            )
        return "success"

    assert retry_call(
        operation,
        attempts=3,
        base_delay=0.25,
        sleep=delays.append,
        random_source=lambda: 0,
    ) == "success"
    assert len(calls) == 3
    assert delays == [0.25, 0.5]


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        (parameter, value)
        for parameter in ("base_delay", "jitter", "max_delay")
        for value in (float("nan"), float("inf"), float("-inf"))
    ],
)
def test_retry_call_rejects_non_finite_delay_parameters(parameter, value):
    from evaluation.providers.base import retry_call

    calls = []

    with pytest.raises(ValueError):
        retry_call(
            lambda: calls.append("called"),
            **{parameter: value},
        )

    assert calls == []


def test_retry_call_raises_final_exception_after_attempt_limit():
    from evaluation.providers.base import ProviderRequestError, retry_call

    errors = [
        ProviderRequestError("test-provider", "first", status_code=500),
        ProviderRequestError("test-provider", "final", status_code=503),
    ]

    def operation():
        raise errors.pop(0)

    with pytest.raises(ProviderRequestError, match="final"):
        retry_call(
            operation,
            attempts=2,
            base_delay=0,
            sleep=lambda _: None,
            random_source=lambda: 0,
        )

    assert errors == []


@pytest.mark.parametrize("status_code", [408, 409, 429, 500, 503, 599])
def test_retry_call_retries_only_transient_http_statuses(status_code):
    from evaluation.providers.base import ProviderRequestError, retry_call

    calls = []

    def operation():
        calls.append("called")
        if len(calls) == 1:
            raise ProviderRequestError(
                "test-provider",
                "transient",
                status_code=status_code,
            )
        return "success"

    assert retry_call(
        operation,
        attempts=2,
        base_delay=0,
        sleep=lambda _: None,
        random_source=lambda: 0,
    ) == "success"
    assert len(calls) == 2


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422])
def test_retry_call_does_not_retry_permanent_http_statuses(status_code):
    from evaluation.providers.base import ProviderRequestError, retry_call

    calls = []

    def operation():
        calls.append("called")
        raise ProviderRequestError(
            "test-provider",
            "permanent",
            status_code=status_code,
        )

    with pytest.raises(ProviderRequestError, match="permanent"):
        retry_call(
            operation,
            attempts=3,
            base_delay=0,
            sleep=lambda _: None,
            random_source=lambda: 0,
        )

    assert len(calls) == 1


@pytest.mark.parametrize("error_type", [TimeoutError, ConnectionError])
def test_retry_call_retries_well_defined_connection_errors(error_type):
    from evaluation.providers.base import retry_call

    calls = []

    def operation():
        calls.append("called")
        if len(calls) == 1:
            raise error_type("temporary transport failure")
        return "success"

    assert retry_call(
        operation,
        attempts=2,
        base_delay=0,
        sleep=lambda _: None,
        random_source=lambda: 0,
    ) == "success"
    assert len(calls) == 2


def test_retry_call_does_not_retry_validation_or_response_errors():
    from evaluation.providers.base import ProviderResponseError, retry_call

    for error in (
        ValueError("invalid request"),
        ProviderResponseError("test-provider", "malformed response"),
    ):
        calls = []

        def operation():
            calls.append("called")
            raise error

        with pytest.raises(type(error), match=str(error)):
            retry_call(
                operation,
                attempts=3,
                base_delay=0,
                sleep=lambda _: None,
                random_source=lambda: 0,
            )
        assert len(calls) == 1


def test_retry_call_honors_retry_after_header_and_bounded_jitter():
    from evaluation.providers.base import retry_call

    class RateLimitError(RuntimeError):
        status_code = 429
        response = SimpleNamespace(headers={"Retry-After": "3"})

    calls = []
    delays = []

    def operation():
        calls.append("called")
        if len(calls) == 1:
            raise RateLimitError("rate limited")
        return "success"

    assert retry_call(
        operation,
        attempts=2,
        base_delay=0.25,
        sleep=delays.append,
        random_source=lambda: 1,
        jitter=0.1,
    ) == "success"
    assert calls == ["called", "called"]
    assert delays == pytest.approx([3.3])


def test_retry_call_caps_huge_retry_after_after_jitter():
    from evaluation.providers.base import ProviderRequestError, retry_call

    calls = []
    delays = []

    def operation():
        calls.append("called")
        if len(calls) == 1:
            raise ProviderRequestError(
                "test-provider",
                "slow down",
                status_code=429,
                retry_after=10_000,
            )
        return "success"

    assert retry_call(
        operation,
        attempts=2,
        base_delay=1,
        max_delay=5,
        jitter=0.5,
        random_source=lambda: 1,
        sleep=delays.append,
    ) == "success"
    assert delays == [5]


@pytest.mark.parametrize("retry_after", [float("inf"), float("nan")])
def test_non_finite_retry_after_is_rejected(retry_after):
    from evaluation.providers.base import ProviderRequestError

    error = ProviderRequestError(
        "test-provider",
        "invalid retry-after",
        status_code=429,
        retry_after=retry_after,
    )

    assert error.retry_after is None


def test_retry_call_clamps_backoff_after_jitter():
    from evaluation.providers.base import ProviderRequestError, retry_call

    calls = []
    delays = []

    def operation():
        calls.append("called")
        if len(calls) == 1:
            raise ProviderRequestError(
                "test-provider",
                "temporary",
                status_code=503,
            )
        return "success"

    assert retry_call(
        operation,
        attempts=2,
        base_delay=4,
        max_delay=5,
        jitter=0.5,
        random_source=lambda: 1,
        sleep=delays.append,
    ) == "success"
    assert delays == [5]


def test_provider_wraps_permanent_request_failure_once_with_context(
    monkeypatch,
):
    from evaluation.providers import create_inference_provider
    from evaluation.providers.base import ProviderRequestError

    _clear_provider_env(monkeypatch)
    _set_env(
        monkeypatch,
        REQUIRED_ENV_BY_PROVIDER[("inference", "azure-openai")],
    )
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        raise ValueError("invalid image detail")

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        )
    )
    provider = create_inference_provider(
        "azure-openai",
        client=client,
        attempts=3,
        base_delay=0,
    )

    with pytest.raises(
        ProviderRequestError,
        match="azure-openai.*invalid image detail",
    ):
        provider.generate("Question", [_frame()])

    assert len(calls) == 1


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (SimpleNamespace(choices=[]), "choices"),
        (
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=None,
                            refusal=None,
                        )
                    )
                ]
            ),
            "content",
        ),
        (
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=None,
                            refusal="blocked by safety policy",
                        )
                    )
                ]
            ),
            "refus",
        ),
    ],
)
@pytest.mark.parametrize("provider_name", ["azure-openai", "openai", "together"])
def test_chat_adapters_raise_controlled_response_errors(
    monkeypatch,
    provider_name,
    response,
    expected,
):
    from evaluation.providers import (
        create_inference_provider,
        create_judge_provider,
    )
    from evaluation.providers.base import ProviderResponseError

    _clear_provider_env(monkeypatch)
    operation = "inference" if provider_name == "azure-openai" else "judge"
    _set_env(
        monkeypatch,
        REQUIRED_ENV_BY_PROVIDER[(operation, provider_name)],
    )
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return response

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        )
    )
    if operation == "inference":
        provider = create_inference_provider(provider_name, client=client)
        invoke = lambda: provider.generate("Question", [_frame()])
    else:
        provider = create_judge_provider(provider_name, client=client)
        invoke = lambda: provider.complete("System", "Prompt", False)

    with pytest.raises(
        ProviderResponseError,
        match=f"{provider_name}.*{expected}",
    ):
        invoke()

    assert len(calls) == 1


def test_azure_wraps_malformed_answer_as_response_error(monkeypatch):
    from evaluation.providers import create_inference_provider
    from evaluation.providers.base import ProviderResponseError

    _clear_provider_env(monkeypatch)
    _set_env(
        monkeypatch,
        REQUIRED_ENV_BY_PROVIDER[("inference", "azure-openai")],
    )
    client = _chat_client('```json\n{"Answer":\n```')
    provider = create_inference_provider("azure-openai", client=client)

    with pytest.raises(
        ProviderResponseError,
        match="azure-openai.*JSON",
    ):
        provider.generate("Question", [_frame()])

    assert len(client.recorder.calls) == 1


def test_anthropic_rejects_response_without_text_block(monkeypatch):
    from evaluation.providers import create_inference_provider
    from evaluation.providers.base import ProviderResponseError

    _clear_provider_env(monkeypatch)
    _set_env(
        monkeypatch,
        REQUIRED_ENV_BY_PROVIDER[("inference", "anthropic")],
    )
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="tool_use")]
        )

    client = SimpleNamespace(
        messages=SimpleNamespace(create=create),
    )
    provider = create_inference_provider("anthropic", client=client)

    with pytest.raises(
        ProviderResponseError,
        match="anthropic.*text",
    ):
        provider.generate("Question", [_frame()])

    assert len(calls) == 1


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(text=""),
        SimpleNamespace(
            text=None,
            prompt_feedback=SimpleNamespace(block_reason="SAFETY"),
        ),
    ],
)
def test_gemini_rejects_empty_or_safety_blocked_response(
    monkeypatch,
    response,
):
    from evaluation.providers import create_inference_provider
    from evaluation.providers.base import ProviderResponseError

    _clear_provider_env(monkeypatch)
    _set_env(
        monkeypatch,
        REQUIRED_ENV_BY_PROVIDER[("inference", "gemini")],
    )
    calls = []

    def generate_content(**kwargs):
        calls.append(kwargs)
        return response

    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=generate_content),
    )
    provider = create_inference_provider("gemini", client=client)

    with pytest.raises(
        ProviderResponseError,
        match="gemini",
    ):
        provider.generate("Question", [_frame()])

    assert len(calls) == 1


@pytest.mark.parametrize(
    "response",
    [
        {"output": {"choices": []}},
        {
            "output": {
                "choices": [
                    {"message": {"content": []}}
                ]
            }
        },
    ],
)
def test_qwen_rejects_missing_choices_or_content(monkeypatch, response):
    from evaluation.providers import create_inference_provider
    from evaluation.providers.base import ProviderResponseError

    _clear_provider_env(monkeypatch)
    _set_env(
        monkeypatch,
        REQUIRED_ENV_BY_PROVIDER[("inference", "qwen")],
    )
    calls = []

    def call(**kwargs):
        calls.append(kwargs)
        return response

    client = SimpleNamespace(
        MultiModalConversation=SimpleNamespace(call=call),
    )
    provider = create_inference_provider("qwen", client=client)

    with pytest.raises(
        ProviderResponseError,
        match="qwen",
    ):
        provider.generate("Question", [_frame()])

    assert len(calls) == 1


@pytest.mark.parametrize(
    ("operation", "provider_name", "module_name", "constructor_name"),
    [
        ("inference", "azure-openai", "openai", "AzureOpenAI"),
        ("judge", "openai", "openai", "OpenAI"),
        ("inference", "anthropic", "anthropic", "Anthropic"),
        ("judge", "together", "together", "Together"),
    ],
)
def test_sdk_client_construction_disables_internal_retries(
    monkeypatch,
    operation,
    provider_name,
    module_name,
    constructor_name,
):
    from evaluation.providers import (
        create_inference_provider,
        create_judge_provider,
    )

    _clear_provider_env(monkeypatch)
    _set_env(
        monkeypatch,
        REQUIRED_ENV_BY_PROVIDER[(operation, provider_name)],
    )
    constructor_calls = []

    def constructor(**kwargs):
        constructor_calls.append(kwargs)
        return object()

    fake_module = SimpleNamespace()
    setattr(fake_module, constructor_name, constructor)
    monkeypatch.setitem(sys.modules, module_name, fake_module)
    factory = (
        create_inference_provider
        if operation == "inference"
        else create_judge_provider
    )
    provider = factory(provider_name)

    provider._get_client()

    assert constructor_calls[0]["max_retries"] == 0


@pytest.mark.parametrize(
    ("provider_name", "module_name", "constructor_name"),
    [
        ("openai", "openai", "OpenAI"),
        ("together", "together", "Together"),
    ],
)
def test_judge_provider_initializes_one_client_without_serializing_requests(
    monkeypatch,
    provider_name,
    module_name,
    constructor_name,
):
    from evaluation.providers import create_judge_provider

    _clear_provider_env(monkeypatch)
    _set_env(
        monkeypatch,
        REQUIRED_ENV_BY_PROVIDER[("judge", provider_name)],
    )
    constructor_calls = []
    request_calls = []
    request_barrier = threading.Barrier(2)

    class FakeConnectionError(Exception):
        pass

    class FakeTimeoutError(Exception):
        pass

    def create(**kwargs):
        request_calls.append((threading.get_ident(), kwargs))
        request_barrier.wait(timeout=1)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"Score": 4}')
                )
            ]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)
        )
    )

    def constructor(**kwargs):
        constructor_calls.append(kwargs)
        time.sleep(0.03)
        return client

    fake_module = SimpleNamespace(
        APIConnectionError=FakeConnectionError,
        APITimeoutError=FakeTimeoutError,
    )
    setattr(fake_module, constructor_name, constructor)
    monkeypatch.setitem(sys.modules, module_name, fake_module)
    provider = create_judge_provider(provider_name)
    start_barrier = threading.Barrier(3)

    def complete(index):
        start_barrier.wait(timeout=1)
        return provider.complete("system", f"prompt-{index}", True)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(complete, index) for index in range(2)]
        start_barrier.wait(timeout=1)
        assert [future.result(timeout=2) for future in futures] == [
            '{"Score": 4}',
            '{"Score": 4}',
        ]

    assert len(constructor_calls) == 1
    assert len(request_calls) == 2
    assert len({thread_id for thread_id, _ in request_calls}) == 2
    assert provider._client is client
    assert provider._transient_exceptions == (
        FakeConnectionError,
        FakeTimeoutError,
    )


def test_judge_provider_protocol_requires_thread_safe_implementations():
    from evaluation.providers.base import JudgeProvider

    assert "thread-safe" in JudgeProvider.complete.__doc__


@pytest.mark.parametrize(
    ("module_name", "sdk_root"),
    [
        ("evaluation.providers.openai", "openai"),
        ("evaluation.providers.anthropic", "anthropic"),
        ("evaluation.providers.gemini", "google"),
        ("evaluation.providers.qwen", "dashscope"),
        ("evaluation.providers.together", "together"),
    ],
)
def test_adapter_module_import_does_not_import_sdk(
    monkeypatch,
    module_name,
    sdk_root,
):
    imported_sdk = []
    real_import = __import__

    def tracking_import(name, *args, **kwargs):
        if name == sdk_root or name.startswith(f"{sdk_root}."):
            imported_sdk.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", tracking_import)
    sys.modules.pop(module_name, None)

    importlib.import_module(module_name)

    assert imported_sdk == []

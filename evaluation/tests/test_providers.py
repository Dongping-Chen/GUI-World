from __future__ import annotations

import base64
import importlib
import io
import sys
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
    from evaluation.providers.base import retry_call

    calls = []
    delays = []

    def operation():
        calls.append("called")
        if len(calls) < 3:
            raise RuntimeError(f"failure-{len(calls)}")
        return "success"

    assert retry_call(
        operation,
        attempts=3,
        base_delay=0.25,
        sleep=delays.append,
    ) == "success"
    assert len(calls) == 3
    assert delays == [0.25, 0.5]


def test_retry_call_raises_final_exception_after_attempt_limit():
    from evaluation.providers.base import retry_call

    errors = [RuntimeError("first"), RuntimeError("final")]

    def operation():
        raise errors.pop(0)

    with pytest.raises(RuntimeError, match="final"):
        retry_call(operation, attempts=2, base_delay=0, sleep=lambda _: None)

    assert errors == []


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

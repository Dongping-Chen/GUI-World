from __future__ import annotations

import copy
import json
import threading
import time
from pathlib import Path

import pytest

from evaluation.jsonl import read_jsonl
from evaluation.prompts import TASK_NAMES, build_tasks


class FakeJudgeProvider:
    def __init__(self, response=None, failures=(), delays=None):
        self.response = response
        self.failures = tuple(failures)
        self.delays = delays or {}
        self.calls = []
        self._lock = threading.Lock()

    def complete(self, system_prompt, prompt, json_output):
        with self._lock:
            call_number = len(self.calls)
            self.calls.append(
                {
                    "system_prompt": system_prompt,
                    "prompt": prompt,
                    "json_output": json_output,
                    "thread": threading.get_ident(),
                }
            )
        for marker, delay in self.delays.items():
            if marker in prompt:
                time.sleep(delay)
                break
        if any(marker in prompt for marker in self.failures):
            raise RuntimeError("simulated provider failure")
        if callable(self.response):
            return self.response(system_prompt, prompt, json_output)
        if self.response is not None:
            return self.response
        if json_output:
            return json.dumps({"Evaluation": "Relevant and accurate.", "Score": 4})
        return " Yes. "


def _write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def _benchmark_result(template, identifier):
    record = {
        "id": identifier,
        "video_path": f"videos/{identifier}.mp4",
        "result": build_tasks(template),
    }
    for task in record["result"].values():
        task["MLLM_result"] = {"Answer": f"model answer for {identifier}"}
    return record


def _judged_result(record):
    value = copy.deepcopy(record)
    for task_name, task in value["result"].items():
        if task_name in {"MCQA", "Reasoning"}:
            task["LLM_Judge"] = "Yes."
        else:
            task["LLM_Judge"] = {
                "Evaluation": "Existing valid evaluation.",
                "Score": 4,
            }
    return value


def test_judge_free_form_uses_rubric_and_returns_validated_object():
    from evaluation.judge import judge_free_form

    provider = FakeJudgeProvider(
        response=json.dumps(
            {"Evaluation": "The response matches.", "Score": 5}
        )
    )

    result = judge_free_form(
        provider,
        "What happened?",
        "The user opened Settings.",
        "Settings was opened.",
    )

    assert result == {
        "Evaluation": "The response matches.",
        "Score": 5,
    }
    call = provider.calls[0]
    assert call["json_output"] is True
    assert "1 to 5" in call["system_prompt"]
    assert "Evaluation" in call["system_prompt"]
    assert "Score" in call["system_prompt"]
    assert "What happened?" in call["prompt"]
    assert "The user opened Settings." in call["prompt"]
    assert "Settings was opened." in call["prompt"]


@pytest.mark.parametrize(
    "response",
    [
        "[]",
        '"not an object"',
        "{}",
        '{"Evaluation": "", "Score": 3}',
        '{"Evaluation": "   ", "Score": 3}',
        '{"Evaluation": 7, "Score": 3}',
        '{"Evaluation": "ok"}',
        '{"Evaluation": "ok", "Score": true}',
        '{"Evaluation": "ok", "Score": 1.0}',
        '{"Evaluation": "ok", "Score": 0}',
        '{"Evaluation": "ok", "Score": 6}',
    ],
)
def test_judge_free_form_rejects_invalid_schema(response):
    from evaluation.judge import JudgeResponseError, judge_free_form

    with pytest.raises(JudgeResponseError):
        judge_free_form(
            FakeJudgeProvider(response=response),
            "question",
            "gold",
            "answer",
        )


@pytest.mark.parametrize(
    "response",
    [
        "Yes",
        " no ",
        "Yes.",
        "No!",
        "**Yes**",
        "（No）",
        "`Yes`",
        "\u3000Yes\u00a0",
    ],
)
def test_judge_multiple_choice_accepts_unambiguous_yes_no_and_keeps_raw(
    response,
):
    from evaluation.judge import judge_multiple_choice

    provider = FakeJudgeProvider(response=response)

    assert judge_multiple_choice(
        provider,
        "Which option?",
        "B",
        "B. Save",
    ) == response
    call = provider.calls[0]
    assert call["json_output"] is False
    assert '"Yes" or "No"' in call["system_prompt"]


@pytest.mark.parametrize(
    "response",
    [
        "",
        "Maybe",
        "Yes because it matches",
        "No, the answer is B",
        "Yes/No",
        "Yes No",
        "affirmative",
        1,
    ],
)
def test_judge_multiple_choice_rejects_ambiguous_or_non_string_output(
    response,
):
    from evaluation.judge import JudgeResponseError, judge_multiple_choice

    with pytest.raises(JudgeResponseError):
        judge_multiple_choice(
            FakeJudgeProvider(response=response),
            "question",
            "gold",
            "answer",
        )


def test_run_judge_routes_task_types_and_skips_missing_model_results(
    tmp_path,
    benchmark_record,
):
    from evaluation.judge import run_judge

    record = _benchmark_result(benchmark_record, "one")
    record["result"]["Caption"].pop("MLLM_result")
    input_path = _write_jsonl(tmp_path / "input.jsonl", [record])
    output_path = tmp_path / "output.jsonl"
    provider = FakeJudgeProvider()

    run_judge(input_path, output_path, provider, workers=2)

    assert len(provider.calls) == 9
    assert sum(call["json_output"] for call in provider.calls) == 7
    output = read_jsonl(output_path)
    assert len(output) == 1
    result = output[0]["result"]
    assert "LLM_Judge" not in result["Caption"]
    assert "judge_error" not in result["Caption"]
    for name in {"MCQA", "Reasoning"}:
        assert result[name]["LLM_Judge"] == " Yes. "
    for name in set(TASK_NAMES) - {"MCQA", "Reasoning", "Caption"}:
        assert result[name]["LLM_Judge"] == {
            "Evaluation": "Relevant and accurate.",
            "Score": 4,
        }


@pytest.mark.parametrize("failure_kind", ["malformed", "provider"])
def test_run_judge_records_task_errors_without_stale_judgment(
    tmp_path,
    benchmark_record,
    failure_kind,
):
    from evaluation.judge import run_judge

    record = _benchmark_result(benchmark_record, "one")
    marker = record["result"]["Prediction"]["q"]
    if failure_kind == "malformed":
        def response(system_prompt, prompt, json_output):
            if marker in prompt:
                return '{"Evaluation": "bad", "Score": true}'
            if json_output:
                return '{"Evaluation": "valid", "Score": 4}'
            return "Yes"

        provider = FakeJudgeProvider(response=response)
    else:
        provider = FakeJudgeProvider(failures={marker})
    input_path = _write_jsonl(tmp_path / "input.jsonl", [record])
    output_path = tmp_path / "output.jsonl"

    run_judge(input_path, output_path, provider)

    prediction = read_jsonl(output_path)[0]["result"]["Prediction"]
    assert "judge_error" in prediction
    assert "LLM_Judge" not in prediction
    assert marker in provider.calls[1]["prompt"]


def test_resume_reuses_only_valid_matching_judgments(
    tmp_path,
    benchmark_record,
):
    from evaluation.judge import run_judge

    current = _benchmark_result(benchmark_record, "resume")
    previous = _judged_result(current)
    previous["result"]["Prediction"]["q"] = "stale question"
    previous["result"]["Description1"]["a"] = "stale gold answer"
    previous["result"]["Caption"]["judge_error"] = "old judge failure"
    previous["result"]["Caption"].pop("LLM_Judge")
    previous["result"]["Reasoning"]["LLM_Judge"] = "Maybe"
    previous["result"]["MCQA"]["MLLM_result"] = {"Answer": "stale model answer"}
    input_path = _write_jsonl(tmp_path / "input.jsonl", [current])
    output_path = _write_jsonl(tmp_path / "output.jsonl", [previous])
    provider = FakeJudgeProvider()

    run_judge(
        input_path,
        output_path,
        provider,
        workers=2,
        resume=True,
    )

    assert len(provider.calls) == 5
    called_prompts = "\n".join(call["prompt"] for call in provider.calls)
    for name in ("Prediction", "Description1", "Caption", "Reasoning", "MCQA"):
        assert current["result"][name]["q"] in called_prompts
    assert current["result"]["Sequential-QA"]["q"] not in called_prompts
    output = read_jsonl(output_path)
    assert len(output) == 1
    for task in output[0]["result"].values():
        assert "LLM_Judge" in task
        assert "judge_error" not in task


def test_complete_resume_makes_no_provider_calls(
    tmp_path,
    benchmark_record,
):
    from evaluation.judge import run_judge

    current = _benchmark_result(benchmark_record, "complete")
    input_path = _write_jsonl(tmp_path / "input.jsonl", [current])
    output_path = _write_jsonl(
        tmp_path / "output.jsonl",
        [_judged_result(current)],
    )
    provider = FakeJudgeProvider()

    run_judge(
        input_path,
        output_path,
        provider,
        resume=True,
    )

    assert provider.calls == []
    assert len(read_jsonl(output_path)) == 1


def test_resume_with_gapped_existing_output_restores_input_order(
    tmp_path,
    benchmark_record,
):
    from evaluation.judge import run_judge

    records = [
        _benchmark_result(benchmark_record, f"id-{index}")
        for index in range(3)
    ]
    input_path = _write_jsonl(tmp_path / "input.jsonl", records)
    output_path = _write_jsonl(
        tmp_path / "output.jsonl",
        [_judged_result(records[2])],
    )

    run_judge(
        input_path,
        output_path,
        FakeJudgeProvider(),
        workers=2,
        resume=True,
    )

    assert [record["id"] for record in read_jsonl(output_path)] == [
        "id-0",
        "id-1",
        "id-2",
    ]


def test_concurrent_records_persist_on_caller_thread_in_input_order(
    tmp_path,
    benchmark_record,
    monkeypatch,
):
    from evaluation import judge
    from evaluation.judge import run_judge

    records = [
        _benchmark_result(benchmark_record, f"id-{index}")
        for index in range(3)
    ]
    for record in records:
        for task in record["result"].values():
            task["q"] = f"{record['id']}: {task['q']}"
    input_path = _write_jsonl(tmp_path / "input.jsonl", records)
    output_path = tmp_path / "output.jsonl"
    provider = FakeJudgeProvider(
        delays={"id-0:": 0.006, "id-1:": 0.003, "id-2:": 0.0}
    )
    caller_thread = threading.get_ident()
    writes = []
    real_write = judge.atomic_write_jsonl

    def tracking_write(path, values):
        snapshot = list(values)
        writes.append(
            (threading.get_ident(), [value["id"] for value in snapshot])
        )
        real_write(path, snapshot)

    monkeypatch.setattr(judge, "atomic_write_jsonl", tracking_write)

    run_judge(
        input_path,
        output_path,
        provider,
        workers=3,
    )

    assert [record["id"] for record in read_jsonl(output_path)] == [
        "id-0",
        "id-1",
        "id-2",
    ]
    assert writes == [
        (caller_thread, ["id-0"]),
        (caller_thread, ["id-0", "id-1"]),
        (caller_thread, ["id-0", "id-1", "id-2"]),
    ]
    assert provider.calls
    assert all(call["thread"] != caller_thread for call in provider.calls)


def test_start_end_select_exact_records(
    tmp_path,
    benchmark_record,
):
    from evaluation.judge import run_judge

    records = [
        _benchmark_result(benchmark_record, f"id-{index}")
        for index in range(4)
    ]
    input_path = _write_jsonl(tmp_path / "input.jsonl", records)
    output_path = tmp_path / "output.jsonl"

    run_judge(
        input_path,
        output_path,
        FakeJudgeProvider(),
        start=1,
        end=3,
        workers=2,
    )

    assert [record["id"] for record in read_jsonl(output_path)] == [
        "id-1",
        "id-2",
    ]


@pytest.mark.parametrize(
    ("start", "end", "workers", "message"),
    [
        (-1, 1, 1, "start"),
        (0, 0, 1, "less than end"),
        (1, 1, 1, "less than end"),
        (0, 3, 1, "record count"),
        (0, 1, 0, "workers"),
        (0, 1, True, "workers"),
    ],
)
def test_run_judge_rejects_invalid_range_and_worker_values(
    tmp_path,
    benchmark_record,
    start,
    end,
    workers,
    message,
):
    from evaluation.judge import JudgeError, run_judge

    input_path = _write_jsonl(
        tmp_path / "input.jsonl",
        [_benchmark_result(benchmark_record, "one")],
    )

    with pytest.raises(JudgeError, match=message):
        run_judge(
            input_path,
            tmp_path / "output.jsonl",
            FakeJudgeProvider(),
            start=start,
            end=end,
            workers=workers,
        )


def test_output_parent_preflight_precedes_provider_calls(
    tmp_path,
    benchmark_record,
):
    from evaluation.judge import JudgeError, run_judge

    input_path = _write_jsonl(
        tmp_path / "input.jsonl",
        [_benchmark_result(benchmark_record, "one")],
    )
    provider = FakeJudgeProvider()
    output_path = tmp_path / "missing" / "output.jsonl"

    with pytest.raises(JudgeError, match=str(output_path.parent)):
        run_judge(input_path, output_path, provider)

    assert provider.calls == []
    assert not output_path.parent.exists()


def test_existing_output_is_preserved_without_resume(
    tmp_path,
    benchmark_record,
):
    from evaluation.judge import JudgeError, run_judge

    input_path = _write_jsonl(
        tmp_path / "input.jsonl",
        [_benchmark_result(benchmark_record, "one")],
    )
    output_path = tmp_path / "output.jsonl"
    original = b'{"id":"existing"}\n'
    output_path.write_bytes(original)
    provider = FakeJudgeProvider()

    with pytest.raises(JudgeError, match=r"already exists.*--resume"):
        run_judge(input_path, output_path, provider)

    assert output_path.read_bytes() == original
    assert provider.calls == []


def test_input_and_output_must_be_different_even_when_resuming(
    tmp_path,
    benchmark_record,
):
    from evaluation.judge import JudgeError, run_judge

    path = _write_jsonl(
        tmp_path / "records.jsonl",
        [_benchmark_result(benchmark_record, "one")],
    )

    with pytest.raises(JudgeError, match="different files"):
        run_judge(
            path,
            path,
            FakeJudgeProvider(),
            resume=True,
        )


def test_cli_defaults_to_auto_output_in_input_parent(
    tmp_path,
    benchmark_record,
    monkeypatch,
):
    from evaluation import judge

    input_directory = tmp_path / "input-data"
    input_directory.mkdir()
    input_path = _write_jsonl(
        input_directory / "benchmark-results.jsonl",
        [_benchmark_result(benchmark_record, "one")],
    )
    different_cwd = tmp_path / "cwd"
    different_cwd.mkdir()
    provider = object()
    factory_calls = []
    run_calls = []
    monkeypatch.setattr(
        judge,
        "create_judge_provider",
        lambda name, **kwargs: (
            factory_calls.append((name, kwargs)) or provider
        ),
    )
    monkeypatch.setattr(
        judge,
        "run_judge",
        lambda *args, **kwargs: run_calls.append((args, kwargs)),
    )
    monkeypatch.chdir(different_cwd)

    assert judge.main(
        [
            "--input",
            str(input_path),
            "--provider",
            "openai",
            "--model-id",
            "judge-model",
            "--start",
            "0",
            "--end",
            "1",
            "--workers",
            "3",
            "--attempts",
            "4",
            "--base-delay",
            "0.5",
            "--resume",
        ]
    ) == 0

    assert factory_calls == [
        (
            "openai",
            {
                "model_id": "judge-model",
                "attempts": 4,
                "base_delay": 0.5,
            },
        )
    ]
    args, kwargs = run_calls[0]
    assert args == (
        input_path,
        input_directory
        / "llm_judge_openai_benchmark-results_0_1.jsonl",
        provider,
    )
    assert kwargs == {
        "start": 0,
        "end": 1,
        "workers": 3,
        "resume": True,
    }


@pytest.mark.parametrize(
    ("provider_name", "canonical"),
    [
        ("openai", "openai"),
        ("together", "together"),
        ("gpt-4", "openai"),
        ("llama3", "together"),
    ],
)
def test_cli_accepts_canonical_and_legacy_provider_names(
    tmp_path,
    benchmark_record,
    monkeypatch,
    provider_name,
    canonical,
):
    from evaluation import judge

    input_path = _write_jsonl(
        tmp_path / "input.jsonl",
        [_benchmark_result(benchmark_record, "one")],
    )
    provider = object()
    factory_names = []
    run_calls = []
    monkeypatch.setattr(
        judge,
        "create_judge_provider",
        lambda name, **kwargs: factory_names.append(name) or provider,
    )
    monkeypatch.setattr(
        judge,
        "run_judge",
        lambda *args, **kwargs: run_calls.append((args, kwargs)),
    )

    assert judge.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(tmp_path / "output.jsonl"),
            "--model",
            provider_name,
        ]
    ) == 0

    assert factory_names == [provider_name]
    assert run_calls[0][0][2] is provider
    assert judge._canonical_judge_provider(provider_name) == canonical


def test_cli_help_documents_judge_options(capsys):
    from evaluation import judge

    with pytest.raises(SystemExit) as raised:
        judge.main(["--help"])

    assert raised.value.code == 0
    help_text = capsys.readouterr().out
    for option in (
        "--provider",
        "--model-id",
        "--start",
        "--end",
        "--workers",
        "--resume",
        "--attempts",
        "--base-delay",
    ):
        assert option in help_text

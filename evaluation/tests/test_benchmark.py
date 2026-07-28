from __future__ import annotations

import argparse
import copy
import json
import tempfile
import warnings
from pathlib import Path

import cv2
import numpy as np
import pytest

from evaluation.jsonl import read_jsonl
from evaluation.prompts import TASK_NAMES


class FakeProvider:
    def __init__(self, failures=()):
        self.calls = []
        self.failures = set(failures)

    def generate(self, prompt, frames):
        call_number = len(self.calls)
        self.calls.append((prompt, frames))
        if call_number in self.failures:
            raise RuntimeError(f"simulated failure {call_number}")
        return {"Answer": f"answer {call_number}"}


def _record(template, identifier, video_path):
    record = copy.deepcopy(template)
    record["id"] = identifier
    record["video_path"] = video_path
    return record


def _write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def _touch_videos(root, records):
    for record in records:
        path = root / record["video_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def _fake_extractor(calls):
    def extract(video_path, strategy, keyframes=()):
        calls.append((video_path, strategy, tuple(keyframes)))
        return [f"frame:{video_path}"]

    return extract


def _complete_result(record, answer_prefix="saved"):
    from evaluation.prompts import build_tasks

    result = build_tasks(record)
    for index, task in enumerate(result.values()):
        task["MLLM_result"] = {"Answer": f"{answer_prefix}-{index}"}
    return {
        "id": record["id"],
        "video_path": record["video_path"],
        "result": result,
    }


def test_run_benchmark_resolves_video_root_extracts_once_and_writes_schema(
    tmp_path,
    benchmark_record,
):
    from evaluation.benchmark import run_benchmark

    record = _record(
        benchmark_record,
        "website-7",
        "final_videos/website/7.mp4",
    )
    input_path = _write_jsonl(tmp_path / "input.jsonl", [record])
    video_root = tmp_path / "videos"
    _touch_videos(video_root, [record])
    extraction_calls = []
    provider = FakeProvider()
    output_path = tmp_path / "output.jsonl"

    run_benchmark(
        input_path,
        output_path,
        provider,
        "website",
        "linspace",
        video_root=video_root,
        frame_extractor=_fake_extractor(extraction_calls),
    )

    assert extraction_calls == [
        (
            video_root / record["video_path"],
            "linspace",
            (),
        )
    ]
    assert len(provider.calls) == 10
    output = read_jsonl(output_path)
    assert len(output) == 1
    assert tuple(output[0]) == ("id", "video_path", "result")
    assert output[0]["id"] == "website-7"
    assert output[0]["video_path"] == record["video_path"]
    assert tuple(output[0]["result"]) == TASK_NAMES
    for task in output[0]["result"].values():
        assert tuple(task) == ("q", "a", "MLLM_result")
        assert task["MLLM_result"]["Answer"].startswith("answer ")
        assert "error" not in task


def test_run_benchmark_uses_annotated_keyframes_once(
    tmp_path,
    benchmark_record,
):
    from evaluation.benchmark import run_benchmark

    record = _record(benchmark_record, "ios-1", "ios/1.mp4")
    record["keyframes"] = [{"frame": 7}, {"frame": 2}]
    input_path = _write_jsonl(tmp_path / "input.jsonl", [record])
    video_root = tmp_path / "videos"
    _touch_videos(video_root, [record])
    calls = []

    run_benchmark(
        input_path,
        tmp_path / "output.jsonl",
        FakeProvider(),
        "ios",
        "annotated",
        video_root=video_root,
        frame_extractor=_fake_extractor(calls),
    )

    assert calls == [
        (
            video_root / "ios/1.mp4",
            "annotated",
            ({"frame": 7}, {"frame": 2}),
        )
    ]


def test_run_benchmark_records_task_failure_without_a_model_result(
    tmp_path,
    benchmark_record,
):
    from evaluation.benchmark import run_benchmark

    record = _record(benchmark_record, "failed", "failed.mp4")
    input_path = _write_jsonl(tmp_path / "input.jsonl", [record])
    (tmp_path / "failed.mp4").touch()
    provider = FakeProvider(failures={1})
    output_path = tmp_path / "output.jsonl"

    run_benchmark(
        input_path,
        output_path,
        provider,
        "website",
        "linspace",
        video_root=tmp_path,
        frame_extractor=_fake_extractor([]),
    )

    result = read_jsonl(output_path)[0]["result"]
    assert "MLLM_result" in result["Sequential-QA"]
    assert result["Prediction"]["error"] == (
        "RuntimeError: simulated failure 1"
    )
    assert "MLLM_result" not in result["Prediction"]
    assert len(provider.calls) == 10


def test_resume_reuses_successes_retries_incomplete_and_clears_error(
    tmp_path,
    benchmark_record,
):
    from evaluation.benchmark import run_benchmark
    from evaluation.prompts import build_tasks

    record = _record(benchmark_record, "resume", "resume.mp4")
    input_path = _write_jsonl(tmp_path / "input.jsonl", [record])
    (tmp_path / "resume.mp4").touch()
    tasks = build_tasks(record)
    tasks["Sequential-QA"]["MLLM_result"] = {"Answer": "keep me"}
    tasks["Prediction"]["error"] = "old failure"
    existing = {
        "id": "resume",
        "video_path": "resume.mp4",
        "result": {
            "Sequential-QA": tasks["Sequential-QA"],
            "Prediction": tasks["Prediction"],
        },
    }
    output_path = _write_jsonl(tmp_path / "output.jsonl", [existing])
    provider = FakeProvider()
    extraction_calls = []

    run_benchmark(
        input_path,
        output_path,
        provider,
        "website",
        "linspace",
        video_root=tmp_path,
        resume=True,
        frame_extractor=_fake_extractor(extraction_calls),
    )

    assert len(provider.calls) == 9
    assert len(extraction_calls) == 1
    output = read_jsonl(output_path)
    assert len(output) == 1
    result = output[0]["result"]
    assert result["Sequential-QA"]["MLLM_result"] == {"Answer": "keep me"}
    assert result["Prediction"]["MLLM_result"] == {"Answer": "answer 0"}
    assert "error" not in result["Prediction"]
    assert all("MLLM_result" in result[name] for name in TASK_NAMES)


def test_failed_resume_retry_removes_stale_result_and_retains_new_error(
    tmp_path,
    benchmark_record,
):
    from evaluation.benchmark import run_benchmark

    record = _record(benchmark_record, "stale", "stale.mp4")
    input_path = _write_jsonl(tmp_path / "input.jsonl", [record])
    (tmp_path / "stale.mp4").touch()
    existing = _complete_result(record)
    existing["result"]["Prediction"]["error"] = "stale failure"
    output_path = _write_jsonl(tmp_path / "output.jsonl", [existing])

    run_benchmark(
        input_path,
        output_path,
        FakeProvider(failures={0}),
        "website",
        "linspace",
        video_root=tmp_path,
        resume=True,
        frame_extractor=_fake_extractor([]),
    )

    prediction = read_jsonl(output_path)[0]["result"]["Prediction"]
    assert prediction["error"] == "RuntimeError: simulated failure 0"
    assert "MLLM_result" not in prediction


def test_complete_resume_makes_no_provider_or_frame_calls(
    tmp_path,
    benchmark_record,
):
    from evaluation.benchmark import run_benchmark

    record = _record(benchmark_record, "complete", "complete.mp4")
    input_path = _write_jsonl(tmp_path / "input.jsonl", [record])
    output_path = _write_jsonl(
        tmp_path / "output.jsonl",
        [_complete_result(record)],
    )
    provider = FakeProvider()

    def unexpected_extraction(*args, **kwargs):
        raise AssertionError("complete resume must not extract frames")

    run_benchmark(
        input_path,
        output_path,
        provider,
        "website",
        "linspace",
        video_root=tmp_path,
        resume=True,
        frame_extractor=unexpected_extraction,
    )

    assert provider.calls == []
    assert len(read_jsonl(output_path)) == 1


@pytest.mark.parametrize(
    "invalid_success",
    [
        "string-result",
        "missing-answer",
        "empty-answer",
        "non-string-answer",
        "changed-question",
        "changed-answer",
    ],
)
def test_resume_retries_results_with_invalid_schema_or_stale_task(
    tmp_path,
    benchmark_record,
    invalid_success,
):
    from evaluation.benchmark import run_benchmark

    record = _record(benchmark_record, "invalid-resume", "resume.mp4")
    input_path = _write_jsonl(tmp_path / "input.jsonl", [record])
    (tmp_path / "resume.mp4").touch()
    existing = _complete_result(record)
    task = existing["result"]["Sequential-QA"]
    if invalid_success == "string-result":
        task["MLLM_result"] = "old raw response"
    elif invalid_success == "missing-answer":
        task["MLLM_result"] = {"Description": "No answer"}
    elif invalid_success == "empty-answer":
        task["MLLM_result"] = {"Answer": "   "}
    elif invalid_success == "non-string-answer":
        task["MLLM_result"] = {"Answer": 7}
    elif invalid_success == "changed-question":
        task["q"] = "An outdated question"
    else:
        task["a"] = "An outdated reference answer"
    output_path = _write_jsonl(tmp_path / "output.jsonl", [existing])
    provider = FakeProvider()
    extraction_calls = []

    run_benchmark(
        input_path,
        output_path,
        provider,
        "website",
        "linspace",
        video_root=tmp_path,
        resume=True,
        frame_extractor=_fake_extractor(extraction_calls),
    )

    assert len(extraction_calls) == 1
    assert len(provider.calls) == 1
    result = read_jsonl(output_path)[0]["result"]
    assert result["Sequential-QA"]["MLLM_result"] == {"Answer": "answer 0"}
    assert result["Prediction"]["MLLM_result"] == {"Answer": "saved-1"}


def test_start_end_select_exact_records_and_atomically_write_each_record(
    tmp_path,
    benchmark_record,
    monkeypatch,
):
    from evaluation import benchmark
    from evaluation.benchmark import run_benchmark

    records = [
        _record(benchmark_record, f"id-{index}", f"{index}.mp4")
        for index in range(4)
    ]
    input_path = _write_jsonl(tmp_path / "input.jsonl", records)
    _touch_videos(tmp_path, records)
    writes = []
    real_atomic_write = benchmark.atomic_write_jsonl

    def tracking_write(path, values):
        snapshot = list(values)
        writes.append([value["id"] for value in snapshot])
        real_atomic_write(path, snapshot)

    monkeypatch.setattr(benchmark, "atomic_write_jsonl", tracking_write)
    output_path = tmp_path / "output.jsonl"

    run_benchmark(
        input_path,
        output_path,
        FakeProvider(),
        "website",
        "linspace",
        video_root=tmp_path,
        start=1,
        end=3,
        frame_extractor=_fake_extractor([]),
    )

    assert writes == [["id-1"], ["id-1", "id-2"]]
    assert [record["id"] for record in read_jsonl(output_path)] == [
        "id-1",
        "id-2",
    ]


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        (-1, 1, "start"),
        (0, 0, "less than end"),
        (2, 1, "less than end"),
        (0, 3, "record count"),
    ],
)
def test_run_benchmark_rejects_invalid_ranges(
    tmp_path,
    benchmark_record,
    start,
    end,
    message,
):
    from evaluation.benchmark import BenchmarkError, run_benchmark

    records = [
        _record(benchmark_record, f"id-{index}", f"{index}.mp4")
        for index in range(2)
    ]
    input_path = _write_jsonl(tmp_path / "input.jsonl", records)

    with pytest.raises(BenchmarkError, match=message):
        run_benchmark(
            input_path,
            tmp_path / "output.jsonl",
            FakeProvider(),
            "website",
            "linspace",
            start=start,
            end=end,
            frame_extractor=_fake_extractor([]),
        )


def test_run_benchmark_rejects_existing_output_without_resume(
    tmp_path,
    benchmark_record,
):
    from evaluation.benchmark import BenchmarkError, run_benchmark

    record = _record(benchmark_record, "one", "one.mp4")
    input_path = _write_jsonl(tmp_path / "input.jsonl", [record])
    output_path = tmp_path / "output.jsonl"
    original = b'{"id": "existing"}\n'
    output_path.write_bytes(original)
    provider = FakeProvider()
    extraction_calls = []

    with pytest.raises(BenchmarkError, match=r"already exists.*--resume"):
        run_benchmark(
            input_path,
            output_path,
            provider,
            "website",
            "linspace",
            frame_extractor=_fake_extractor(extraction_calls),
        )

    assert output_path.read_bytes() == original
    assert extraction_calls == []
    assert provider.calls == []


@pytest.mark.parametrize("parent_kind", ["missing", "file"])
def test_output_parent_preflight_fails_before_frame_or_provider_calls(
    tmp_path,
    benchmark_record,
    parent_kind,
):
    from evaluation.benchmark import BenchmarkError, run_benchmark

    record = _record(benchmark_record, "one", "one.mp4")
    input_path = _write_jsonl(tmp_path / "input.jsonl", [record])
    (tmp_path / "one.mp4").touch()
    parent = tmp_path / "invalid-output-parent"
    if parent_kind == "file":
        parent.write_text("not a directory", encoding="utf-8")
    output_path = parent / "output.jsonl"
    provider = FakeProvider()
    extraction_calls = []

    with pytest.raises(BenchmarkError) as raised:
        run_benchmark(
            input_path,
            output_path,
            provider,
            "website",
            "linspace",
            video_root=tmp_path,
            frame_extractor=_fake_extractor(extraction_calls),
        )

    assert extraction_calls == []
    assert provider.calls == []
    assert str(parent) in str(raised.value)
    if parent_kind == "missing":
        assert not parent.exists()


def test_output_preflight_temp_creation_failure_precedes_paid_calls(
    tmp_path,
    benchmark_record,
    monkeypatch,
):
    from evaluation.benchmark import BenchmarkError, run_benchmark

    record = _record(benchmark_record, "one", "one.mp4")
    input_path = _write_jsonl(tmp_path / "input.jsonl", [record])
    (tmp_path / "one.mp4").touch()
    provider = FakeProvider()
    extraction_calls = []

    def deny_temp_file(*args, **kwargs):
        raise PermissionError("simulated unwritable directory")

    monkeypatch.setattr(
        tempfile,
        "NamedTemporaryFile",
        deny_temp_file,
    )

    with pytest.raises(
        BenchmarkError,
        match="simulated unwritable directory",
    ):
        run_benchmark(
            input_path,
            tmp_path / "output.jsonl",
            provider,
            "website",
            "linspace",
            video_root=tmp_path,
            frame_extractor=_fake_extractor(extraction_calls),
        )

    assert extraction_calls == []
    assert provider.calls == []
    assert not (tmp_path / "output.jsonl").exists()


@pytest.mark.parametrize("warnings_as_errors", [False, True])
def test_output_preflight_cleanup_does_not_mask_primary_close_failure(
    tmp_path,
    benchmark_record,
    monkeypatch,
    warnings_as_errors,
):
    from evaluation.benchmark import BenchmarkError, run_benchmark

    record = _record(benchmark_record, "one", "one.mp4")
    input_path = _write_jsonl(tmp_path / "input.jsonl", [record])
    (tmp_path / "one.mp4").touch()
    temporary_path = tmp_path / ".output.jsonl.preflight.tmp"
    provider = FakeProvider()
    extraction_calls = []

    class CloseFailure:
        name = str(temporary_path)

        def __enter__(self):
            temporary_path.touch()
            return self

        def __exit__(self, error_type, error, traceback):
            raise OSError("primary probe close failure")

    def fail_unlink(path):
        raise OSError("secondary probe cleanup failure")

    monkeypatch.setattr(
        tempfile,
        "NamedTemporaryFile",
        lambda **kwargs: CloseFailure(),
    )
    monkeypatch.setattr(Path, "unlink", fail_unlink)

    def invoke():
        return run_benchmark(
            input_path,
            tmp_path / "output.jsonl",
            provider,
            "website",
            "linspace",
            video_root=tmp_path,
            frame_extractor=_fake_extractor(extraction_calls),
        )

    if warnings_as_errors:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with pytest.raises(
                BenchmarkError,
                match="primary probe close failure",
            ) as raised:
                invoke()
    else:
        with pytest.warns(
            RuntimeWarning,
            match="secondary probe cleanup failure",
        ):
            with pytest.raises(
                BenchmarkError,
                match="primary probe close failure",
            ) as raised:
                invoke()

    assert any(
        "secondary probe cleanup failure" in detail
        for detail in raised.value.cleanup_errors
    )
    assert extraction_calls == []
    assert provider.calls == []


def test_output_preflight_probe_leaves_no_file_before_extraction(
    tmp_path,
    benchmark_record,
):
    from evaluation.benchmark import run_benchmark

    record = _record(benchmark_record, "one", "one.mp4")
    input_path = _write_jsonl(tmp_path / "input.jsonl", [record])
    (tmp_path / "one.mp4").touch()

    def inspect_directory(video_path, strategy, keyframes=()):
        assert not list(tmp_path.glob(".output.jsonl.*.tmp"))
        return ["frame"]

    run_benchmark(
        input_path,
        tmp_path / "output.jsonl",
        FakeProvider(),
        "website",
        "linspace",
        video_root=tmp_path,
        frame_extractor=inspect_directory,
    )

    assert not list(tmp_path.glob(".output.jsonl.*.tmp"))


def test_run_benchmark_rejects_same_input_and_output_even_in_resume_mode(
    tmp_path,
    benchmark_record,
):
    from evaluation.benchmark import BenchmarkError, run_benchmark

    path = _write_jsonl(
        tmp_path / "records.jsonl",
        [_record(benchmark_record, "one", "one.mp4")],
    )

    with pytest.raises(BenchmarkError, match="different files"):
        run_benchmark(
            path,
            path,
            FakeProvider(),
            "website",
            "linspace",
            resume=True,
            frame_extractor=_fake_extractor([]),
        )


def test_missing_video_error_names_record_and_resolved_path(
    tmp_path,
    benchmark_record,
):
    from evaluation.benchmark import BenchmarkError, run_benchmark

    record = _record(benchmark_record, "missing-id", "nested/missing.mp4")
    input_path = _write_jsonl(tmp_path / "input.jsonl", [record])
    missing = tmp_path / "videos" / "nested/missing.mp4"

    with pytest.raises(BenchmarkError) as raised:
        run_benchmark(
            input_path,
            tmp_path / "output.jsonl",
            FakeProvider(),
            "website",
            "linspace",
            video_root=tmp_path / "videos",
            frame_extractor=_fake_extractor([]),
        )

    assert "missing-id" in str(raised.value)
    assert str(missing) in str(raised.value)


def test_real_tiny_video_path_runs_with_offline_provider(
    tmp_path,
    benchmark_record,
):
    from evaluation.benchmark import run_benchmark

    record = _record(benchmark_record, "tiny", "tiny.avi")
    input_path = _write_jsonl(tmp_path / "input.jsonl", [record])
    writer = cv2.VideoWriter(
        str(tmp_path / "tiny.avi"),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10,
        (16, 12),
    )
    assert writer.isOpened()
    try:
        writer.write(np.zeros((12, 16, 3), dtype=np.uint8))
        writer.write(np.full((12, 16, 3), 200, dtype=np.uint8))
    finally:
        writer.release()
    provider = FakeProvider()

    run_benchmark(
        input_path,
        tmp_path / "output.jsonl",
        provider,
        "website",
        "linspace",
        video_root=tmp_path,
    )

    assert len(provider.calls) == 10
    assert all(len(frames) == 2 for _, frames in provider.calls)


def test_cli_canonical_options_and_auto_output_are_deterministic(
    tmp_path,
    benchmark_record,
    monkeypatch,
):
    from evaluation import benchmark

    input_directory = tmp_path / "benchmark-data"
    input_directory.mkdir()
    input_path = _write_jsonl(
        input_directory / "input.jsonl",
        [_record(benchmark_record, "one", "one.mp4")],
    )
    working_directory = tmp_path / "different-working-directory"
    working_directory.mkdir()
    provider = object()
    factory_calls = []
    run_calls = []

    def fake_factory(name, **options):
        factory_calls.append((name, options))
        return provider

    monkeypatch.setattr(benchmark, "create_inference_provider", fake_factory)
    monkeypatch.setattr(
        benchmark,
        "run_benchmark",
        lambda *args, **kwargs: run_calls.append((args, kwargs)),
    )
    monkeypatch.chdir(working_directory)

    assert benchmark.main(
        [
            "--input",
            str(input_path),
            "--output",
            "auto",
            "--provider",
            "azure-openai",
            "--scenario",
            "ios",
            "--strategy",
            "linspace",
            "--model-id",
            "deployment",
            "--video-root",
            str(tmp_path / "videos"),
            "--start",
            "0",
            "--end",
            "1",
            "--attempts",
            "4",
            "--base-delay",
            "0.25",
            "--resume",
        ]
    ) == 0

    assert factory_calls == [
        (
            "azure-openai",
            {
                "model_id": "deployment",
                "attempts": 4,
                "base_delay": 0.25,
            },
        )
    ]
    args, kwargs = run_calls[0]
    assert args[:3] == (
        input_path,
        input_directory / "output_azure-openai_ios_linspace_0_1.jsonl",
        provider,
    )
    assert args[3:] == ("ios", "linspace")
    assert kwargs == {
        "video_root": tmp_path / "videos",
        "start": 0,
        "end": 1,
        "resume": True,
    }


def test_cli_omitting_output_uses_auto_filename(
    tmp_path,
    benchmark_record,
    monkeypatch,
):
    from evaluation import benchmark

    input_path = _write_jsonl(
        tmp_path / "input.jsonl",
        [_record(benchmark_record, "one", "one.mp4")],
    )
    provider = object()
    run_calls = []
    monkeypatch.setattr(
        benchmark,
        "create_inference_provider",
        lambda name, **kwargs: provider,
    )
    monkeypatch.setattr(
        benchmark,
        "run_benchmark",
        lambda *args, **kwargs: run_calls.append((args, kwargs)),
    )

    assert benchmark.main(
        [
            "--input",
            str(input_path),
            "--provider",
            "gemini",
            "--scenario",
            "website",
            "--strategy",
            "annotated",
        ]
    ) == 0

    assert run_calls[0][0][1] == (
        tmp_path / "output_gemini_website_annotated_0_1.jsonl"
    )


def test_cli_legacy_option_names_and_values_map_to_canonical_workflow(
    tmp_path,
    benchmark_record,
    monkeypatch,
):
    from evaluation import benchmark

    input_path = _write_jsonl(
        tmp_path / "input.jsonl",
        [_record(benchmark_record, "one", "one.mp4")],
    )
    provider = object()
    factory_names = []
    run_calls = []
    monkeypatch.setattr(
        benchmark,
        "create_inference_provider",
        lambda name, **kwargs: factory_names.append(name) or provider,
    )
    monkeypatch.setattr(
        benchmark,
        "run_benchmark",
        lambda *args, **kwargs: run_calls.append((args, kwargs)),
    )

    assert benchmark.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(tmp_path / "legacy.jsonl"),
            "--model",
            "GPT-4V",
            "--setting",
            "IOS",
            "--keyframe",
            "Random",
        ]
    ) == 0

    assert factory_names == ["GPT-4V"]
    args, kwargs = run_calls[0]
    assert args[3:] == ("ios", "linspace")
    assert kwargs["start"] == 0
    assert kwargs["end"] is None
    assert kwargs["resume"] is False


def test_cli_help_documents_workflow_options(capsys):
    from evaluation import benchmark

    with pytest.raises(SystemExit) as raised:
        benchmark.main(["--help"])

    assert raised.value.code == 0
    help_text = capsys.readouterr().out
    for option in (
        "--provider",
        "--scenario",
        "--strategy",
        "--model-id",
        "--video-root",
        "--start",
        "--end",
        "--resume",
        "--attempts",
        "--base-delay",
    ):
        assert option in help_text


@pytest.mark.parametrize(
    "value",
    ["nan", "NaN", "inf", "+inf", "-inf", "Infinity", "-Infinity"],
)
def test_non_negative_float_rejects_non_finite_values(value):
    from evaluation.benchmark import _non_negative_float

    with pytest.raises(argparse.ArgumentTypeError, match="non-negative"):
        _non_negative_float(value)


def test_cli_reports_invalid_strategy_without_a_traceback(
    tmp_path,
    benchmark_record,
    capsys,
):
    from evaluation import benchmark

    input_path = _write_jsonl(
        tmp_path / "input.jsonl",
        [_record(benchmark_record, "one", "one.mp4")],
    )

    with pytest.raises(SystemExit) as raised:
        benchmark.main(
            [
                "--input",
                str(input_path),
                "--output",
                str(tmp_path / "output.jsonl"),
                "--provider",
                "gemini",
                "--scenario",
                "website",
                "--strategy",
                "unsupported",
            ]
        )

    assert raised.value.code == 2
    error_text = capsys.readouterr().err
    assert "Unknown frame strategy" in error_text
    assert "Traceback" not in error_text

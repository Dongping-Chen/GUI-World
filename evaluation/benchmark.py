from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path
from typing import Callable, Sequence

from PIL import Image

from evaluation.frames import (
    FrameExtractionError,
    canonical_strategy,
    extract_frames,
)
from evaluation.jsonl import (
    JsonlError,
    _report_cleanup_failure,
    atomic_write_jsonl,
    read_jsonl,
    record_id,
    upsert_record,
)
from evaluation.prompts import (
    TASK_NAMES,
    build_prompt,
    build_tasks,
    canonical_scenario,
)
from evaluation.providers import (
    InferenceProvider,
    ProviderError,
    _canonical_provider,
    create_inference_provider,
)


class BenchmarkError(RuntimeError):
    """Raised when benchmark orchestration cannot proceed safely."""


FrameExtractor = Callable[
    [Path, str, Sequence[dict]],
    list[Image.Image],
]


def _validate_range(
    record_count: int,
    start: int,
    end: int | None,
) -> int:
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise BenchmarkError("Range start must be a non-negative integer.")
    if end is None:
        selected_end = record_count
    elif isinstance(end, bool) or not isinstance(end, int):
        raise BenchmarkError("Range end must be an integer.")
    else:
        selected_end = end
    if start >= selected_end:
        raise BenchmarkError(
            f"Range start ({start}) must be less than end ({selected_end})."
        )
    if selected_end > record_count:
        raise BenchmarkError(
            f"Range end ({selected_end}) exceeds input record count "
            f"({record_count})."
        )
    return selected_end


def _resolved_video_path(record: dict, video_root: Path | None) -> Path:
    try:
        value = record["video_path"]
    except KeyError as error:
        raise BenchmarkError(
            f"Record {record_id(record)!r} is missing video_path."
        ) from error
    path = Path(str(value))
    if video_root is not None and not path.is_absolute():
        path = video_root / path
    return path


def _preflight_output(path: Path) -> None:
    parent = path.parent
    if not parent.exists():
        raise BenchmarkError(
            f"Output directory does not exist: {parent}"
        )
    if not parent.is_dir():
        raise BenchmarkError(
            f"Output parent is not a directory: {parent}"
        )

    temporary_path: Path | None = None
    primary_error: BaseException | None = None
    try:
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=parent,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
        except OSError as error:
            raise BenchmarkError(
                f"Could not create an atomic temporary output file in "
                f"{parent}: {error}"
            ) from error
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError as error:
                if primary_error is not None:
                    _report_cleanup_failure(
                        primary_error,
                        f"Output preflight cleanup failed ({error}) for "
                        f"{temporary_path} while handling "
                        f"{type(primary_error).__name__}.",
                    )
                else:
                    raise BenchmarkError(
                        f"Could not remove output preflight file "
                        f"{temporary_path}: {error}"
                    ) from error


def _successful_task(value, current_task: dict) -> bool:
    result = (
        value.get("MLLM_result")
        if isinstance(value, dict)
        else None
    )
    answer = result.get("Answer") if isinstance(result, dict) else None
    return (
        isinstance(value, dict)
        and value.get("q") == current_task["q"]
        and value.get("a") == current_task["a"]
        and isinstance(answer, str)
        and bool(answer.strip())
        and "error" not in value
    )


def _task_error(error: Exception) -> str:
    message = str(error).strip()
    if message:
        return f"{type(error).__name__}: {message}"
    return type(error).__name__


def _result_record(
    source: dict,
    tasks: dict[str, dict],
) -> dict:
    identifier = (
        source["id"]
        if "id" in source
        else source["video_path"]
    )
    return {
        "id": identifier,
        "video_path": source["video_path"],
        "result": tasks,
    }


def run_benchmark(
    input_path: Path,
    output_path: Path,
    provider: InferenceProvider,
    scenario: str,
    strategy: str,
    video_root: Path | None = None,
    start: int = 0,
    end: int | None = None,
    resume: bool = False,
    *,
    frame_extractor: FrameExtractor = extract_frames,
) -> None:
    """Run inference and atomically upsert one output record at a time."""

    input_path = Path(input_path)
    output_path = Path(output_path)
    video_root = None if video_root is None else Path(video_root)
    if input_path.resolve() == output_path.resolve():
        raise BenchmarkError(
            "Input and output must be different files; benchmark results "
            "cannot overwrite the source JSONL."
        )
    if output_path.exists() and not resume:
        raise BenchmarkError(
            f"Output file already exists: {output_path}. Use --resume to "
            "continue it or choose a different output."
        )
    _preflight_output(output_path)

    canonical_scenario_name = canonical_scenario(scenario)
    canonical_strategy_name = canonical_strategy(strategy)
    input_records = read_jsonl(input_path)
    selected_end = _validate_range(len(input_records), start, end)
    output_records = (
        read_jsonl(output_path)
        if resume and output_path.exists()
        else []
    )
    existing_by_id = {
        record_id(record): record
        for record in output_records
    }

    for source in input_records[start:selected_end]:
        stable_id = record_id(source)
        try:
            tasks = build_tasks(source)
        except (KeyError, TypeError, ValueError) as error:
            raise BenchmarkError(
                f"Could not build tasks for record {stable_id!r}: {error}"
            ) from error

        previous = existing_by_id.get(stable_id)
        previous_tasks = (
            previous.get("result", {})
            if isinstance(previous, dict)
            else {}
        )
        pending = []
        for task_name in TASK_NAMES:
            task = tasks[task_name]
            old_task = (
                previous_tasks.get(task_name)
                if isinstance(previous_tasks, dict)
                else None
            )
            if _successful_task(old_task, task):
                task["MLLM_result"] = old_task["MLLM_result"]
                continue
            if isinstance(old_task, dict):
                if "error" in old_task:
                    task["error"] = old_task["error"]
                task.pop("MLLM_result", None)
            pending.append(task_name)

        if pending:
            video_path = _resolved_video_path(source, video_root)
            if not video_path.is_file():
                raise BenchmarkError(
                    f"Video for record {stable_id!r} does not exist: "
                    f"{video_path}"
                )
            try:
                frames = frame_extractor(
                    video_path,
                    canonical_strategy_name,
                    source.get("keyframes", ()),
                )
            except Exception as error:
                raise BenchmarkError(
                    f"Could not extract frames for record {stable_id!r} "
                    f"from {video_path}: {error}"
                ) from error

            for task_name in pending:
                task = tasks[task_name]
                prompt = build_prompt(
                    canonical_scenario_name,
                    task_name,
                    task["q"],
                )
                try:
                    response = provider.generate(prompt, frames)
                    if not isinstance(response, dict):
                        raise TypeError(
                            "provider result must be a JSON object"
                        )
                except Exception as error:
                    task.pop("MLLM_result", None)
                    task["error"] = _task_error(error)
                else:
                    task["MLLM_result"] = response
                    task.pop("error", None)

        value = _result_record(source, tasks)
        output_records = upsert_record(output_records, value)
        existing_by_id[stable_id] = value
        atomic_write_jsonl(output_path, output_records)


def _positive_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be a positive integer"
        ) from error
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _non_negative_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be a non-negative number"
        ) from error
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be a non-negative number")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run GUI-World benchmark inference.",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        default="auto",
        help="Output JSONL path, or 'auto' for a deterministic filename.",
    )
    parser.add_argument(
        "--provider",
        "--model",
        dest="provider",
        required=True,
        help=(
            "Inference provider: azure-openai, anthropic, gemini, or qwen "
            "(legacy --model names are accepted)."
        ),
    )
    parser.add_argument(
        "--scenario",
        "--setting",
        dest="scenario",
        required=True,
        help=(
            "Scenario: android, ios, multi, software, website, or xr "
            "(legacy --setting names are accepted)."
        ),
    )
    parser.add_argument(
        "--strategy",
        "--keyframe",
        dest="strategy",
        required=True,
        help=(
            "Frame strategy: linspace or annotated "
            "(legacy --keyframe Random/Extracted are accepted)."
        ),
    )
    parser.add_argument("--model-id")
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--attempts", type=_positive_integer, default=3)
    parser.add_argument("--base-delay", type=_non_negative_float, default=1.0)
    return parser


def _auto_output_path(
    input_path: Path,
    provider: str,
    scenario: str,
    strategy: str,
    start: int,
    end: int | None,
) -> Path:
    records = read_jsonl(input_path)
    selected_end = _validate_range(len(records), start, end)
    provider_name = _canonical_provider(provider)
    filename = (
        f"output_{provider_name}_{scenario}_{strategy}_"
        f"{start}_{selected_end}.jsonl"
    )
    return input_path.parent / filename


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        scenario = canonical_scenario(args.scenario)
        strategy = canonical_strategy(args.strategy)
        if args.output == "auto":
            output_path = _auto_output_path(
                args.input,
                args.provider,
                scenario,
                strategy,
                args.start,
                args.end,
            )
        else:
            output_path = Path(args.output)
        provider = create_inference_provider(
            args.provider,
            model_id=args.model_id,
            attempts=args.attempts,
            base_delay=args.base_delay,
        )
        run_benchmark(
            args.input,
            output_path,
            provider,
            scenario,
            strategy,
            video_root=args.video_root,
            start=args.start,
            end=args.end,
            resume=args.resume,
        )
    except (
        BenchmarkError,
        FrameExtractionError,
        JsonlError,
        ProviderError,
        ValueError,
    ) as error:
        parser.error(str(error))
    print(f"Results saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

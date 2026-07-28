from __future__ import annotations

import argparse
import copy
import json
import math
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from evaluation.jsonl import (
    JsonlError,
    _report_cleanup_failure,
    atomic_write_jsonl,
    read_jsonl,
    record_id,
    upsert_record,
)
from evaluation.prompts import TASK_NAMES
from evaluation.providers import (
    JudgeProvider,
    ProviderError,
    create_judge_provider,
)


class JudgeError(RuntimeError):
    """Raised when judge orchestration cannot proceed safely."""


class JudgeResponseError(ValueError):
    """Raised when a judge response does not match the required schema."""


_FREE_FORM_SYSTEM_PROMPT = """You are an impartial judge. Compare a model
response with a gold-standard answer for the supplied question. Briefly
explain similarities and differences in relevance, factual accuracy, depth,
detail, and how well the response meets the user's need. Then assign a score
from 1 to 5: 1 for irrelevant, fundamentally incorrect, or harmful answers;
2 for harmless but low-quality answers that do not meet the need; 3 for
medium-quality answers that generally meet the need but have notable flaws;
4 for answers close to the reference and strong across the important
dimensions; and 5 only for answers that fully satisfy the need and approach
perfect quality. Be objective and rigorous. Return one JSON object using a
non-empty string field named "Evaluation" and an integer field named "Score".
"""

_MULTIPLE_CHOICE_SYSTEM_PROMPT = """You are an impartial judge for a
multiple-choice question. Compare the model response with the one correct
answer. Output only "Yes" or "No": use "Yes" if they match and "No" if they
do not. Do not include an explanation or any other information.
"""

_YES_NO_EDGE_CHARACTERS = (
    " \t\r\n.,!?;:'\"`()[]{}<>*_~#-—–"
    "，。！？；：、“”‘’（）【】《》"
)

_JUDGE_PROVIDER_ALIASES = {
    "openai": "openai",
    "gpt-4": "openai",
    "together": "together",
    "llama3": "together",
}


def _free_form_prompt(question: str, gold: str, answer: str) -> str:
    return (
        "Here is the response for you to judge:\n"
        f"Question: {question}\n"
        f"Gold-standard Answer: {gold}\n"
        f"Model Response: {answer}\n\n"
        "Return the required JSON object directly."
    )


def _multiple_choice_prompt(
    question: str,
    gold: str,
    answer: str,
) -> str:
    return (
        "Judge this multiple-choice response:\n"
        f"Question: {question}\n"
        f"Correct Answer: {gold}\n"
        f"Model Response: {answer}\n\n"
        'Output only "Yes" or "No".'
    )


def _validate_free_form(value) -> dict:
    if not isinstance(value, dict):
        raise JudgeResponseError(
            "Free-form judge output must be a JSON object."
        )
    evaluation = value.get("Evaluation")
    if not isinstance(evaluation, str) or not evaluation.strip():
        raise JudgeResponseError(
            "Free-form judge Evaluation must be a non-empty string."
        )
    score = value.get("Score")
    if (
        isinstance(score, bool)
        or not isinstance(score, int)
        or not 1 <= score <= 5
    ):
        raise JudgeResponseError(
            "Free-form judge Score must be an integer from 1 through 5."
        )
    return {"Evaluation": evaluation, "Score": score}


def _validate_multiple_choice(value) -> str:
    if not isinstance(value, str):
        raise JudgeResponseError(
            "Multiple-choice judge output must be a string."
        )
    start = 0
    end = len(value)
    while (
        start < end
        and (
            value[start].isspace()
            or value[start] in _YES_NO_EDGE_CHARACTERS
        )
    ):
        start += 1
    while (
        end > start
        and (
            value[end - 1].isspace()
            or value[end - 1] in _YES_NO_EDGE_CHARACTERS
        )
    ):
        end -= 1
    normalized = value[start:end].casefold()
    if normalized not in {"yes", "no"}:
        raise JudgeResponseError(
            'Multiple-choice judge output must be unambiguously "Yes" or "No".'
        )
    return value


def judge_free_form(
    provider: JudgeProvider,
    question: str,
    gold: str,
    answer: str,
) -> dict:
    """Judge a free-form answer and return a validated score object."""

    raw = provider.complete(
        _FREE_FORM_SYSTEM_PROMPT,
        _free_form_prompt(question, gold, answer),
        True,
    )
    if not isinstance(raw, str):
        raise JudgeResponseError(
            "Free-form judge output must be JSON text."
        )
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise JudgeResponseError(
            f"Free-form judge output is malformed JSON: {error.msg}"
        ) from error
    return _validate_free_form(value)


def judge_multiple_choice(
    provider: JudgeProvider,
    question: str,
    gold: str,
    answer: str,
) -> str:
    """Judge a multiple-choice answer while retaining the legacy raw text."""

    raw = provider.complete(
        _MULTIPLE_CHOICE_SYSTEM_PROMPT,
        _multiple_choice_prompt(question, gold, answer),
        False,
    )
    return _validate_multiple_choice(raw)


def _task_error(error: Exception) -> str:
    message = str(error).strip()
    if message:
        return f"{type(error).__name__}: {message}"
    return type(error).__name__


def _model_answer(task: dict) -> str | None:
    result = task.get("MLLM_result")
    if not isinstance(result, dict):
        return None
    answer = result.get("Answer")
    return answer if isinstance(answer, str) else None


def _valid_existing_judge(
    task_name: str,
    previous,
    current: dict,
) -> bool:
    if (
        not isinstance(previous, dict)
        or "judge_error" in previous
        or previous.get("q") != current.get("q")
        or previous.get("a") != current.get("a")
        or previous.get("MLLM_result") != current.get("MLLM_result")
        or "LLM_Judge" not in previous
    ):
        return False
    try:
        if task_name in {"MCQA", "Reasoning"}:
            _validate_multiple_choice(previous["LLM_Judge"])
        else:
            _validate_free_form(previous["LLM_Judge"])
    except JudgeResponseError:
        return False
    return True


def _judge_record(
    source: dict,
    previous,
    provider: JudgeProvider,
) -> dict:
    current = copy.deepcopy(source)
    stable_id = record_id(current)
    tasks = current.get("result")
    if not isinstance(tasks, dict):
        raise JudgeError(
            f"Record {stable_id!r} result must be an object."
        )
    previous_tasks = (
        previous.get("result", {})
        if isinstance(previous, dict)
        else {}
    )

    for task_name in TASK_NAMES:
        task = tasks.get(task_name)
        if not isinstance(task, dict):
            continue
        answer = _model_answer(task)
        if answer is None:
            continue
        task.pop("LLM_Judge", None)
        task.pop("judge_error", None)
        old_task = (
            previous_tasks.get(task_name)
            if isinstance(previous_tasks, dict)
            else None
        )
        if _valid_existing_judge(task_name, old_task, task):
            task["LLM_Judge"] = copy.deepcopy(old_task["LLM_Judge"])
            continue

        try:
            if task_name in {"MCQA", "Reasoning"}:
                judgment = judge_multiple_choice(
                    provider,
                    task["q"],
                    task["a"],
                    answer,
                )
            else:
                judgment = judge_free_form(
                    provider,
                    task["q"],
                    task["a"],
                    answer,
                )
        except Exception as error:
            task.pop("LLM_Judge", None)
            task["judge_error"] = _task_error(error)
        else:
            task["LLM_Judge"] = judgment
            task.pop("judge_error", None)
    return current


def _validate_range(
    record_count: int,
    start: int,
    end: int | None,
) -> int:
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise JudgeError("Range start must be a non-negative integer.")
    if end is None:
        selected_end = record_count
    elif isinstance(end, bool) or not isinstance(end, int):
        raise JudgeError("Range end must be an integer.")
    else:
        selected_end = end
    if start >= selected_end:
        raise JudgeError(
            f"Range start ({start}) must be less than end ({selected_end})."
        )
    if selected_end > record_count:
        raise JudgeError(
            f"Range end ({selected_end}) exceeds input record count "
            f"({record_count})."
        )
    return selected_end


def _validate_workers(workers: int) -> None:
    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or workers < 1
    ):
        raise JudgeError("Judge workers must be a positive integer.")


def _preflight_output(path: Path) -> None:
    parent = path.parent
    if not parent.exists():
        raise JudgeError(f"Output directory does not exist: {parent}")
    if not parent.is_dir():
        raise JudgeError(f"Output parent is not a directory: {parent}")

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
            raise JudgeError(
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
                        f"Judge output preflight cleanup failed ({error}) "
                        f"for {temporary_path} while handling "
                        f"{type(primary_error).__name__}.",
                    )
                else:
                    raise JudgeError(
                        f"Could not remove output preflight file "
                        f"{temporary_path}: {error}"
                    ) from error


def _ordered_existing_records(
    existing: list[dict],
    input_records: list[dict],
) -> list[dict]:
    positions = {
        record_id(record): index
        for index, record in enumerate(input_records)
    }
    fallback = len(input_records)
    indexed = list(enumerate(existing))
    indexed.sort(
        key=lambda value: (
            positions.get(record_id(value[1]), fallback),
            value[0],
        )
    )
    return [record for _, record in indexed]


def run_judge(
    input_path: Path,
    output_path: Path,
    provider: JudgeProvider,
    *,
    start: int = 0,
    end: int | None = None,
    workers: int = 8,
    resume: bool = False,
) -> None:
    """Judge benchmark records concurrently and persist them in input order."""

    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise JudgeError(
            "Input and output must be different files; judged results "
            "cannot overwrite the benchmark source JSONL."
        )
    if output_path.exists() and not resume:
        raise JudgeError(
            f"Output file already exists: {output_path}. Use --resume to "
            "continue it or choose a different output."
        )
    _validate_workers(workers)
    _preflight_output(output_path)

    input_records = read_jsonl(input_path)
    selected_end = _validate_range(len(input_records), start, end)
    output_records = (
        read_jsonl(output_path)
        if resume and output_path.exists()
        else []
    )
    output_records = _ordered_existing_records(
        output_records,
        input_records,
    )
    existing_by_id = {
        record_id(record): record
        for record in output_records
    }
    selected = input_records[start:selected_end]
    completed = {}
    next_to_write = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_indices = {
            executor.submit(
                _judge_record,
                source,
                existing_by_id.get(record_id(source)),
                provider,
            ): index
            for index, source in enumerate(selected)
        }
        for future in as_completed(future_indices):
            index = future_indices[future]
            completed[index] = future.result()
            while next_to_write in completed:
                value = completed.pop(next_to_write)
                output_records = upsert_record(output_records, value)
                output_records = _ordered_existing_records(
                    output_records,
                    input_records,
                )
                existing_by_id[record_id(value)] = value
                atomic_write_jsonl(output_path, output_records)
                next_to_write += 1


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


def _canonical_judge_provider(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Unknown judge provider: {value!r}")
    normalized = value.strip().casefold()
    try:
        return _JUDGE_PROVIDER_ALIASES[normalized]
    except KeyError as error:
        raise ValueError(f"Unknown judge provider: {value!r}") from error


def _auto_output_path(
    input_path: Path,
    provider_name: str,
    start: int,
    end: int | None,
) -> Path:
    records = read_jsonl(input_path)
    selected_end = _validate_range(len(records), start, end)
    canonical = _canonical_judge_provider(provider_name)
    filename = (
        f"llm_judge_{canonical}_{input_path.stem}_"
        f"{start}_{selected_end}.jsonl"
    )
    return input_path.parent / filename


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Judge GUI-World benchmark inference results.",
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
            "Judge provider: openai or together "
            "(legacy gpt-4 and llama3 names are accepted)."
        ),
    )
    parser.add_argument("--model-id")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--workers", type=_positive_integer, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--attempts", type=_positive_integer, default=3)
    parser.add_argument("--base-delay", type=_non_negative_float, default=1.0)
    return parser


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        canonical = _canonical_judge_provider(args.provider)
        if args.output == "auto":
            output_path = _auto_output_path(
                args.input,
                canonical,
                args.start,
                args.end,
            )
        else:
            output_path = Path(args.output)
        provider = create_judge_provider(
            args.provider,
            model_id=args.model_id,
            attempts=args.attempts,
            base_delay=args.base_delay,
        )
        run_judge(
            args.input,
            output_path,
            provider,
            start=args.start,
            end=args.end,
            workers=args.workers,
            resume=args.resume,
        )
    except (JudgeError, JsonlError, ProviderError, ValueError) as error:
        parser.error(str(error))
    print(f"Judged results saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

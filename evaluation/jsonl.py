from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


class JsonlError(RuntimeError):
    """Raised when JSONL input or output is invalid."""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def record_id(record: dict[str, Any]) -> str:
    """Return the stable identifier used for a benchmark record."""

    if "id" in record:
        return str(record["id"])
    if "video_path" in record:
        return str(record["video_path"])
    raise JsonlError("Record must contain either an 'id' or 'video_path' field.")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read strict JSON objects from *path* and reject duplicate record IDs."""

    if not path.exists():
        raise JsonlError(f"JSONL file does not exist: {path}")
    if not path.is_file():
        raise JsonlError(f"JSONL path is not a file: {path}")

    records: list[dict[str, Any]] = []
    id_lines: dict[str, int] = {}
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line, parse_constant=_reject_json_constant)
                except json.JSONDecodeError as error:
                    raise JsonlError(
                        f"Malformed JSON in {path} at line {line_number}: "
                        f"{error.msg}"
                    ) from error
                except ValueError as error:
                    raise JsonlError(
                        f"Malformed JSON in {path} at line {line_number}: {error}"
                    ) from error

                if not isinstance(value, dict):
                    raise JsonlError(
                        f"JSONL row in {path} at line {line_number} "
                        "must be a JSON object."
                    )
                try:
                    stable_id = record_id(value)
                except JsonlError as error:
                    raise JsonlError(
                        f"Invalid record in {path} at line {line_number}: {error}"
                    ) from error
                if stable_id in id_lines:
                    raise JsonlError(
                        f"Duplicate record ID {stable_id!r} in {path} at "
                        f"line {line_number}; first seen at line "
                        f"{id_lines[stable_id]}."
                    )
                id_lines[stable_id] = line_number
                records.append(value)
    except (OSError, UnicodeError) as error:
        raise JsonlError(f"Could not read JSONL file {path}: {error}") from error
    return records


def atomic_write_jsonl(
    path: Path,
    records: Iterable[dict[str, Any]],
) -> None:
    """Atomically replace *path* with UTF-8 JSON objects from *records*."""

    parent = path.parent
    if not parent.exists():
        raise JsonlError(f"Destination directory does not exist: {parent}")
    if not parent.is_dir():
        raise JsonlError(f"Destination parent is not a directory: {parent}")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            for line_number, record in enumerate(records, start=1):
                if not isinstance(record, dict):
                    raise JsonlError(
                        f"JSONL record {line_number} must be a JSON object."
                    )
                try:
                    serialized = json.dumps(
                        record,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                except (TypeError, ValueError) as error:
                    raise JsonlError(
                        f"JSONL record {line_number} is not valid JSON: {error}"
                    ) from error
                temporary.write(serialized)
                temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())

        os.replace(temporary_path, path)
        temporary_path = None
    except JsonlError:
        raise
    except OSError as error:
        raise JsonlError(
            f"Could not atomically write JSONL file {path}: {error}"
        ) from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            except OSError as error:
                raise JsonlError(
                    f"Could not clean temporary JSONL file "
                    f"{temporary_path}: {error}"
                ) from error


def upsert_record(
    records: list[dict[str, Any]],
    value: dict[str, Any],
) -> list[dict[str, Any]]:
    """Replace a matching record in place, or append a new record."""

    value_id = record_id(value)
    updated = list(records)
    for index, record in enumerate(updated):
        if record_id(record) == value_id:
            updated[index] = value
            return updated
    updated.append(value)
    return updated

from __future__ import annotations

import json
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any, Iterable


class JsonlError(RuntimeError):
    """Raised when JSONL input or output is invalid."""


def _report_cleanup_failure(
    primary_error: BaseException,
    message: str,
) -> None:
    try:
        cleanup_errors = getattr(primary_error, "cleanup_errors", None)
        if cleanup_errors is None:
            cleanup_errors = []
            setattr(primary_error, "cleanup_errors", cleanup_errors)
        if isinstance(cleanup_errors, list):
            cleanup_errors.append(message)
    except BaseException:
        pass

    try:
        warnings.warn(
            message,
            RuntimeWarning,
            stacklevel=3,
        )
    except BaseException:
        pass


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def record_id(record: dict[str, Any]) -> str:
    """Return the stable identifier used for a benchmark record."""

    if "id" in record:
        return str(record["id"])
    if "video_path" in record:
        return str(record["video_path"])
    raise JsonlError("Record must contain either an 'id' or 'video_path' field.")


def _fsync_directory(directory: Path) -> None:
    if os.name != "posix":
        return

    descriptor: int | None = None
    primary_error: BaseException | None = None
    try:
        try:
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            descriptor = os.open(directory, flags)
            os.fsync(descriptor)
        except OSError as error:
            raise JsonlError(
                "JSONL replacement may already be committed, but destination "
                f"directory {directory} could not be synced: {error}"
            ) from error
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError as error:
                if primary_error is not None:
                    _report_cleanup_failure(
                        primary_error,
                        f"Directory descriptor cleanup failed ({error}) for "
                        f"{directory} while handling "
                        f"{type(primary_error).__name__}.",
                    )
                else:
                    raise JsonlError(
                        "JSONL replacement may already be committed, but "
                        f"destination directory {directory} could not be "
                        f"closed after syncing: {error}"
                    ) from error


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
    primary_error: BaseException | None = None
    try:
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
                id_lines: dict[str, int] = {}
                for line_number, record in enumerate(records, start=1):
                    if not isinstance(record, dict):
                        raise JsonlError(
                            f"JSONL record {line_number} must be a JSON object."
                        )
                    try:
                        stable_id = record_id(record)
                    except JsonlError as error:
                        raise JsonlError(
                            f"Invalid JSONL record {line_number}: {error}"
                        ) from error
                    if stable_id in id_lines:
                        raise JsonlError(
                            f"Duplicate record ID {stable_id!r} at JSONL "
                            f"record {line_number}; first seen at record "
                            f"{id_lines[stable_id]}."
                        )
                    id_lines[stable_id] = line_number
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
            _fsync_directory(parent)
        except JsonlError:
            raise
        except OSError as error:
            raise JsonlError(
                f"Could not atomically write JSONL file {path}: {error}"
            ) from error
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            except OSError as error:
                if primary_error is not None:
                    _report_cleanup_failure(
                        primary_error,
                        f"Temporary JSONL cleanup failed ({error}) for "
                        f"{temporary_path} while handling "
                        f"{type(primary_error).__name__}.",
                    )
                else:
                    raise JsonlError(
                        f"Could not clean temporary JSONL file "
                        f"{temporary_path}: {error}"
                    ) from error


def upsert_record(
    records: list[dict[str, Any]],
    value: dict[str, Any],
) -> list[dict[str, Any]]:
    """Replace a matching record in place, or append a new record."""

    record_ids: list[str] = []
    id_positions: dict[str, int] = {}
    for position, record in enumerate(records, start=1):
        stable_id = record_id(record)
        if stable_id in id_positions:
            raise JsonlError(
                f"Duplicate record ID {stable_id!r} in existing records at "
                f"position {position}; first seen at position "
                f"{id_positions[stable_id]}."
            )
        id_positions[stable_id] = position
        record_ids.append(stable_id)

    value_id = record_id(value)
    updated = list(records)
    for index, stable_id in enumerate(record_ids):
        if stable_id == value_id:
            updated[index] = value
            return updated
    updated.append(value)
    return updated

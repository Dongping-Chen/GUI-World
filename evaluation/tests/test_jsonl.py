from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation import jsonl
from evaluation.jsonl import (
    JsonlError,
    atomic_write_jsonl,
    read_jsonl,
    record_id,
    upsert_record,
)


def test_read_jsonl_loads_objects_and_ignores_blank_lines(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text(
        '\n{"id": "first", "answer": "保存"}\n   \n{"id": "second"}\n',
        encoding="utf-8",
    )

    assert read_jsonl(path) == [
        {"id": "first", "answer": "保存"},
        {"id": "second"},
    ]


def test_read_jsonl_reports_path_line_and_decoder_message(tmp_path):
    path = tmp_path / "malformed.jsonl"
    malformed_row = '{"id": "broken",}'
    path.write_text(
        f'{{"id": "first"}}\n\n{malformed_row}\n',
        encoding="utf-8",
    )
    with pytest.raises(json.JSONDecodeError) as decoder_error:
        json.loads(malformed_row)

    with pytest.raises(JsonlError) as raised:
        read_jsonl(path)

    message = str(raised.value)
    assert str(path) in message
    assert "line 3" in message
    assert decoder_error.value.msg in message


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_read_jsonl_rejects_non_standard_json_constants(tmp_path, constant):
    path = tmp_path / "non-standard.jsonl"
    path.write_text(f'{{"id": "first", "value": {constant}}}\n', encoding="utf-8")

    with pytest.raises(JsonlError) as raised:
        read_jsonl(path)

    message = str(raised.value)
    assert str(path) in message
    assert "line 1" in message
    assert constant in message


@pytest.mark.parametrize("row", ["[]", '"text"', "null", "1"])
def test_read_jsonl_rejects_non_object_rows(tmp_path, row):
    path = tmp_path / "non-object.jsonl"
    path.write_text(f'{{"id": "first"}}\n{row}\n', encoding="utf-8")

    with pytest.raises(JsonlError) as raised:
        read_jsonl(path)

    message = str(raised.value)
    assert str(path) in message
    assert "line 2" in message
    assert "object" in message


def test_read_jsonl_rejects_duplicate_stable_ids(tmp_path):
    path = tmp_path / "duplicate.jsonl"
    path.write_text(
        '{"video_path": "final_videos/ios/0.mp4"}\n'
        '{"video_path": "final_videos/ios/0.mp4", "answer": "duplicate"}\n',
        encoding="utf-8",
    )

    with pytest.raises(JsonlError) as raised:
        read_jsonl(path)

    message = str(raised.value)
    assert str(path) in message
    assert "line 2" in message
    assert "duplicate" in message.lower()
    assert "final_videos/ios/0.mp4" in message


def test_read_jsonl_rejects_rows_without_a_stable_id(tmp_path):
    path = tmp_path / "missing-id.jsonl"
    path.write_text('{"answer": "unidentifiable"}\n', encoding="utf-8")

    with pytest.raises(JsonlError) as raised:
        read_jsonl(path)

    message = str(raised.value)
    assert str(path) in message
    assert "line 1" in message
    assert "id" in message
    assert "video_path" in message


@pytest.mark.parametrize(
    "scenario",
    ["ios", "xr", "multi", "software", "website"],
)
def test_record_id_falls_back_to_video_path_for_five_scenarios(scenario):
    path = f"final_videos/{scenario}/example.mp4"

    assert record_id({"video_path": path}) == path


@pytest.mark.parametrize(
    ("identifier", "expected"),
    [
        (0, "0"),
        (None, "None"),
    ],
)
def test_record_id_uses_id_key_exactly_even_when_value_is_falsy(
    identifier,
    expected,
):
    assert record_id(
        {
            "id": identifier,
            "video_path": "final_videos/ios/fallback.mp4",
        }
    ) == expected


def test_record_id_reports_missing_id_fields():
    with pytest.raises(JsonlError, match=r"'id'.*'video_path'"):
        record_id({"answer": "missing"})


def test_atomic_write_jsonl_writes_utf8_objects_one_per_line(tmp_path):
    path = tmp_path / "records.jsonl"
    records = [
        {"id": "first", "answer": "保存"},
        {"video_path": "final_videos/xr/0.mp4"},
    ]

    atomic_write_jsonl(path, records)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line) for line in lines] == records
    assert "\\u4fdd" not in path.read_text(encoding="utf-8")


def test_atomic_write_jsonl_rejects_non_object_records_and_cleans_temp(tmp_path):
    path = tmp_path / "records.jsonl"

    with pytest.raises(JsonlError, match="object"):
        atomic_write_jsonl(path, [{"id": "first"}, ["not", "an", "object"]])

    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_atomic_write_jsonl_rejects_non_standard_json_values(tmp_path):
    path = tmp_path / "records.jsonl"

    with pytest.raises(JsonlError, match="JSON"):
        atomic_write_jsonl(path, [{"id": "first", "value": float("nan")}])

    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_atomic_write_jsonl_requires_an_existing_parent_directory(tmp_path):
    path = tmp_path / "missing" / "records.jsonl"

    with pytest.raises(JsonlError) as raised:
        atomic_write_jsonl(path, [{"id": "first"}])

    message = str(raised.value)
    assert str(path.parent) in message
    assert "directory" in message.lower()
    assert not path.exists()


def test_atomic_write_jsonl_fsyncs_directory_after_replace(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "records.jsonl"
    events = []
    real_replace = jsonl.os.replace

    def track_replace(source, destination):
        events.append(("replace", Path(destination)))
        real_replace(source, destination)

    def track_directory_fsync(directory):
        events.append(("fsync", directory))

    monkeypatch.setattr(jsonl.os, "replace", track_replace)
    monkeypatch.setattr(jsonl, "_fsync_directory", track_directory_fsync)

    atomic_write_jsonl(path, [{"id": "first"}])

    assert events == [("replace", path), ("fsync", path.parent)]


def test_fsync_directory_opens_syncs_and_closes_on_posix(tmp_path, monkeypatch):
    calls = []
    descriptor = 73

    monkeypatch.setattr(jsonl.os, "name", "posix")
    monkeypatch.setattr(
        jsonl.os,
        "open",
        lambda path, flags: calls.append(("open", path, flags)) or descriptor,
    )
    monkeypatch.setattr(
        jsonl.os,
        "fsync",
        lambda value: calls.append(("fsync", value)),
    )
    monkeypatch.setattr(
        jsonl.os,
        "close",
        lambda value: calls.append(("close", value)),
    )

    jsonl._fsync_directory(tmp_path)

    expected_flags = jsonl.os.O_RDONLY | getattr(jsonl.os, "O_DIRECTORY", 0)
    assert calls == [
        ("open", tmp_path, expected_flags),
        ("fsync", descriptor),
        ("close", descriptor),
    ]


def test_fsync_directory_is_an_explicit_noop_off_posix(tmp_path, monkeypatch):
    def unexpected_open(path, flags):
        raise AssertionError("Non-POSIX platforms must not open directories.")

    monkeypatch.setattr(jsonl.os, "name", "nt")
    monkeypatch.setattr(jsonl.os, "open", unexpected_open)

    assert jsonl._fsync_directory(tmp_path) is None


@pytest.mark.parametrize("operation", ["open", "fsync", "close"])
def test_fsync_directory_errors_warn_that_replacement_may_be_committed(
    tmp_path,
    monkeypatch,
    operation,
):
    descriptor = 41
    closed = []
    monkeypatch.setattr(jsonl.os, "name", "posix")
    monkeypatch.setattr(jsonl.os, "open", lambda path, flags: descriptor)
    monkeypatch.setattr(jsonl.os, "fsync", lambda value: None)
    monkeypatch.setattr(jsonl.os, "close", closed.append)

    def fail(*args):
        raise OSError(f"simulated {operation} failure")

    monkeypatch.setattr(jsonl.os, operation, fail)

    with pytest.raises(JsonlError) as raised:
        jsonl._fsync_directory(tmp_path)

    message = str(raised.value)
    assert "may already be committed" in message
    assert str(tmp_path) in message
    assert f"simulated {operation} failure" in message
    assert closed == ([descriptor] if operation == "fsync" else [])


def test_atomic_write_jsonl_replace_failure_preserves_output_and_cleans_temp(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "records.jsonl"
    original = b'{"id": "old"}\n'
    path.write_bytes(original)

    def fail_replace(source, destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(jsonl.os, "replace", fail_replace)

    with pytest.raises(JsonlError, match="simulated replace failure"):
        atomic_write_jsonl(path, [{"id": "new"}])

    assert path.read_bytes() == original
    assert [entry for entry in tmp_path.iterdir() if entry != path] == []


def test_cleanup_failure_does_not_mask_replace_failure(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "records.jsonl"
    original = b'{"id": "old"}\n'
    path.write_bytes(original)
    real_unlink = Path.unlink

    def fail_replace(source, destination):
        raise OSError("primary replace failure")

    def fail_unlink(temporary_path):
        raise OSError("secondary cleanup failure")

    monkeypatch.setattr(jsonl.os, "replace", fail_replace)
    monkeypatch.setattr(Path, "unlink", fail_unlink)

    with pytest.warns(RuntimeWarning, match="secondary cleanup failure"):
        with pytest.raises(JsonlError, match="primary replace failure") as raised:
            atomic_write_jsonl(path, [{"id": "new"}])

    assert "secondary cleanup failure" not in str(raised.value)
    assert path.read_bytes() == original
    temporary_paths = [entry for entry in tmp_path.iterdir() if entry != path]
    assert len(temporary_paths) == 1
    real_unlink(temporary_paths[0])


def test_cleanup_failure_does_not_mask_keyboard_interrupt(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "records.jsonl"
    original = b'{"id": "old"}\n'
    path.write_bytes(original)
    real_unlink = Path.unlink

    def interrupted_records():
        yield {"id": "first"}
        raise KeyboardInterrupt("primary interrupt")

    def fail_unlink(temporary_path):
        raise OSError("secondary cleanup failure")

    monkeypatch.setattr(Path, "unlink", fail_unlink)

    with pytest.warns(RuntimeWarning, match=r"secondary cleanup failure.*\.tmp"):
        with pytest.raises(KeyboardInterrupt, match="primary interrupt"):
            atomic_write_jsonl(path, interrupted_records())

    assert path.read_bytes() == original
    temporary_paths = [entry for entry in tmp_path.iterdir() if entry != path]
    assert len(temporary_paths) == 1
    real_unlink(temporary_paths[0])


def test_atomic_write_jsonl_rejects_missing_stable_id_without_replacing_output(
    tmp_path,
):
    path = tmp_path / "records.jsonl"
    original = b'{"id": "old"}\n'
    path.write_bytes(original)

    with pytest.raises(JsonlError, match=r"'id'.*'video_path'"):
        atomic_write_jsonl(path, [{"answer": "missing stable ID"}])

    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_write_jsonl_rejects_coerced_duplicate_ids_before_replace(
    tmp_path,
):
    path = tmp_path / "records.jsonl"
    original = b'{"id": "old"}\n'
    path.write_bytes(original)

    with pytest.raises(JsonlError, match=r"Duplicate.*'1'"):
        atomic_write_jsonl(path, [{"id": 1}, {"id": "1"}])

    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_upsert_record_replaces_matching_id_in_place_without_mutating_input():
    records = [
        {"id": "first", "answer": "old first"},
        {"video_path": "final_videos/ios/0.mp4", "answer": "old second"},
        {"id": "third", "answer": "old third"},
    ]
    replacement = {
        "video_path": "final_videos/ios/0.mp4",
        "answer": "new second",
    }

    updated = upsert_record(records, replacement)

    assert updated == [
        {"id": "first", "answer": "old first"},
        replacement,
        {"id": "third", "answer": "old third"},
    ]
    assert records[1]["answer"] == "old second"


def test_upsert_record_appends_new_id_after_existing_records():
    records = [{"id": "first"}, {"id": "second"}]
    value = {"id": "third"}

    assert upsert_record(records, value) == [*records, value]


def test_upsert_record_rejects_preexisting_coerced_duplicate_ids():
    records = [
        {"id": 1, "answer": "first"},
        {"id": "1", "answer": "duplicate"},
    ]

    with pytest.raises(JsonlError, match=r"Duplicate.*'1'"):
        upsert_record(records, {"id": 1, "answer": "replacement"})

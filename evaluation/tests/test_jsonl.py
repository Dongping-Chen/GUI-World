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

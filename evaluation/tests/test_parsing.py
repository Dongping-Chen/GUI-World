from __future__ import annotations

import pytest

from evaluation.parsing import (
    ResponseParseError,
    extract_json_object,
    normalize_answer,
)


def test_extract_json_object_from_balanced_prose():
    assert extract_json_object(
        'Result follows: {"Description": "The {Save} button", "Answer": "B"} done.'
    ) == {"Description": "The {Save} button", "Answer": "B"}


def test_parse_fenced_answer():
    assert normalize_answer('```json\n{"Answer":"B"}\n```') == {"Answer": "B"}


def test_plain_text_becomes_answer():
    assert normalize_answer("The user clicks Save.") == {
        "Answer": "The user clicks Save."
    }


def test_dictionary_input_is_normalized():
    assert normalize_answer(
        {
            "Description": "The settings page is visible.",
            "Analysis": "Save is the next action.",
            "Answer": "Click Save.",
        }
    ) == {
        "Description": "The settings page is visible.",
        "Analysis": "Save is the next action.",
        "Answer": "Click Save.",
    }


def test_answer_key_is_case_insensitive():
    assert normalize_answer({"ANSWER": "C"}) == {"Answer": "C"}


def test_only_supported_string_fields_are_retained():
    assert normalize_answer(
        {
            "Description": "A dialog is open.",
            "Analysis": ["ignored"],
            "Answer": "Close it.",
            "Confidence": 0.9,
        }
    ) == {"Description": "A dialog is open.", "Answer": "Close it."}


def test_embedded_json_is_normalized():
    assert normalize_answer(
        'Here is the result: {"Analysis": "The icon is selected.", "answer": "A"}'
    ) == {"Analysis": "The icon is selected.", "Answer": "A"}


def test_empty_output_is_rejected():
    with pytest.raises(ResponseParseError, match="empty"):
        normalize_answer(" \n\t")


def test_malformed_fenced_json_is_rejected():
    with pytest.raises(ResponseParseError, match="JSON"):
        normalize_answer('```json\n{"Answer": "B"\n```')


def test_json_dictionary_without_answer_is_rejected():
    with pytest.raises(ResponseParseError, match="Answer"):
        normalize_answer('{"Description": "A toolbar is visible."}')


def test_invalid_answer_type_is_rejected():
    with pytest.raises(ResponseParseError, match="string"):
        normalize_answer('{"Answer": ["B"]}')

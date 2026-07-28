from __future__ import annotations

import pytest

from evaluation.prompts import (
    FREE_FORM_TASKS,
    MULTIPLE_CHOICE_TASKS,
    TASK_NAMES,
    build_prompt,
    build_tasks,
    canonical_scenario,
    normalize_options,
)


EXPECTED_TASK_NAMES = (
    "Sequential-QA",
    "Prediction",
    "Reasoning",
    "Description1",
    "Description2",
    "Caption",
    "static QA",
    "MCQA",
    "Conversation1",
    "Conversation2",
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("android", "android"),
        ("mobile", "android"),
        ("IOS", "ios"),
        ("ios", "ios"),
        ("MULTI", "multi"),
        ("Software", "software"),
        ("WEBSITE", "website"),
        ("XR", "xr"),
    ],
)
def test_canonical_scenario_accepts_canonical_and_legacy_names(value, expected):
    assert canonical_scenario(value) == expected


def test_canonical_scenario_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown scenario"):
        canonical_scenario("desktop")


def test_task_groups_cover_the_exact_output_tasks():
    assert TASK_NAMES == EXPECTED_TASK_NAMES
    assert MULTIPLE_CHOICE_TASKS == frozenset({"MCQA", "Reasoning"})
    assert FREE_FORM_TASKS == frozenset(TASK_NAMES) - MULTIPLE_CHOICE_TASKS


def test_normalize_options_preserves_dictionary_order():
    assert normalize_options(
        {"A": "Alpha", "B": "B. Beta", "C": "Charlie", "D": "D) Delta"}
    ) == ["A. Alpha", "B. Beta", "C. Charlie", "D) Delta"]


def test_normalize_options_preserves_list_order_and_adds_missing_prefixes():
    assert normalize_options(
        ["First", "B. Second", "C) Third", "D. Fourth"]
    ) == ["A. First", "B. Second", "C) Third", "D. Fourth"]


def test_build_tasks_returns_all_ten_tasks_in_order(benchmark_record):
    assert tuple(build_tasks(benchmark_record)) == EXPECTED_TASK_NAMES


def test_build_tasks_retains_questions_and_answers(benchmark_record):
    tasks = build_tasks(benchmark_record)

    assert tasks["Sequential-QA"] == {
        "q": benchmark_record["Sequential-QA"]["Question"],
        "a": benchmark_record["Sequential-QA"]["Answer"],
    }
    assert tasks["Prediction"] == {
        "q": benchmark_record["Prediction"]["Question"],
        "a": benchmark_record["Prediction"]["Answer"],
    }
    assert tasks["static QA"] == {
        "q": benchmark_record["static QA"]["Question"],
        "a": benchmark_record["static QA"]["Answer"],
    }


def test_build_tasks_appends_all_multiple_choice_options(benchmark_record):
    tasks = build_tasks(benchmark_record)

    for option in ("A. A folder", "B. A document", "C. A toolbar", "D. A dialog"):
        assert option in tasks["MCQA"]["q"]
    for option in benchmark_record["Reasoning"]["Options"]:
        assert option in tasks["Reasoning"]["q"]
    assert tasks["MCQA"]["a"] == benchmark_record["MCQA"]["Correct Answer"]
    assert tasks["Reasoning"]["a"] == benchmark_record["Reasoning"]["Correct Answer"]


def test_build_tasks_uses_fixed_description_and_caption_questions(benchmark_record):
    tasks = build_tasks(benchmark_record)

    description_question = (
        "Please provide a detailed description of what occurs throughout these "
        "sequential GUI images, focusing on the changes in the GUI elements or "
        "scenes rather than static aspects of a single frame."
    )
    assert tasks["Description1"] == {
        "q": description_question,
        "a": benchmark_record["Description1"],
    }
    assert tasks["Description2"] == {
        "q": description_question,
        "a": benchmark_record["Description2"],
    }
    assert tasks["Caption"] == {
        "q": (
            "Please give me a concise caption of these sequential GUI images, "
            "focusing on the changes in the GUI elements or scenes rather than "
            "static aspects of a single frame."
        ),
        "a": benchmark_record["Caption"],
    }


def test_build_tasks_includes_round_one_conversation_context(benchmark_record):
    tasks = build_tasks(benchmark_record)

    assert tasks["Conversation1"] == {
        "q": "User: Which item is selected? Assistant:",
        "a": "The document is selected.",
    }
    assert tasks["Conversation2"] == {
        "q": (
            "User: Which item is selected? Assistant: The document is selected. "
            "User: What opens next? Assistant:"
        ),
        "a": "A menu opens next.",
    }


def test_build_prompt_includes_scenario_task_and_json_contract():
    prompt = build_prompt(
        "IOS",
        "Prediction",
        "What will the user likely do next?",
    )

    assert "iOS" in prompt
    assert "predicting the next action" in prompt
    assert "What will the user likely do next?" in prompt
    assert "'Description'" in prompt
    assert "'Analysis'" in prompt
    assert "'Answer'" in prompt
    assert "JSON" in prompt


def test_build_prompt_rejects_unknown_tasks():
    with pytest.raises(ValueError, match="Unknown task"):
        build_prompt("ios", "Unsupported", "What happened?")

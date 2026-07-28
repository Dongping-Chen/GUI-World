from __future__ import annotations

import re


TASK_NAMES = (
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

MULTIPLE_CHOICE_TASKS = frozenset({"MCQA", "Reasoning"})
FREE_FORM_TASKS = frozenset(TASK_NAMES) - MULTIPLE_CHOICE_TASKS

_SCENARIO_ALIASES = {
    "android": "android",
    "mobile": "android",
    "ios": "ios",
    "multi": "multi",
    "software": "software",
    "website": "website",
    "xr": "xr",
}

_SCENARIO_PROMPTS = {
    "android": (
        "You are an AI visual assistant. Here are sequential GUI mobile "
        "interface images selected from a GUI video."
    ),
    "ios": (
        "You are an AI visual assistant. Here are sequential GUI iOS interface "
        "images selected from a GUI video."
    ),
    "multi": (
        "You are an AI visual assistant. Here are sequential GUI interface "
        "images of interactions among multiple software applications and "
        "websites, selected from a GUI video."
    ),
    "software": (
        "You are an AI visual assistant. Here are sequential GUI interface "
        "images of a specific software application, selected from a GUI video."
    ),
    "website": (
        "You are an AI visual assistant. Here are sequential GUI interface "
        "images of a desktop website, selected from a GUI video."
    ),
    "xr": (
        "You are an AI visual assistant. Here are sequential images of mixed "
        "reality combining a GUI interface and the real world, selected from "
        "a GUI video."
    ),
}

_TASK_PROMPTS = {
    "Sequential-QA": (
        "This is a question about sequential information in the sequential "
        "images."
    ),
    "Prediction": (
        "This is a question about predicting the next action based on the "
        "previous actions in the sequential images."
    ),
    "Reasoning": (
        "This is a multiple-choice question with only one correct answer. This "
        "question may need multiple steps of reasoning according to the visual "
        "information in the sequential images."
    ),
    "Description1": (
        "Please give me a detailed description of these sequential images."
    ),
    "Description2": "Offer a thorough analysis of these sequential images.",
    "Caption": "Please give me a concise caption of these sequential images.",
    "static QA": (
        "This is a question about static information such as text, icons, and "
        "layout in the sequential images."
    ),
    "MCQA": (
        "This is a multiple-choice question with only one correct answer. This "
        "question may require sequential analysis of the visual information in "
        "the sequential images."
    ),
    "Conversation1": (
        "Act as an assistant and answer the user's question about these "
        "sequential images."
    ),
    "Conversation2": (
        "This is a multi-turn conversation task. You will be provided with the "
        "first-round conversation and should act as an assistant to answer the "
        "user's second-round question according to these sequential images."
    ),
}

_DESCRIPTION_QUESTION = (
    "Please provide a detailed description of what occurs throughout these "
    "sequential GUI images, focusing on the changes in the GUI elements or "
    "scenes rather than static aspects of a single frame."
)

_CAPTION_QUESTION = (
    "Please give me a concise caption of these sequential GUI images, focusing "
    "on the changes in the GUI elements or scenes rather than static aspects "
    "of a single frame."
)

_NOTICE = (
    "You can first provide an overall description of these sequential images, "
    "and then analyze the user's question according to the sequential images "
    "and description. Finally, give an answer based on this description and "
    "the image information. Please format your output as JSON, with key "
    "'Description' for the description of these sequential images, key "
    "'Analysis' for your analysis of the user's question, and key 'Answer' for "
    "your answer to the user's question."
)

_OPTION_PREFIX = re.compile(r"^\s*(?:[A-Za-z][.):]|\[[A-Za-z]\])\s*")


def canonical_scenario(value: str) -> str:
    """Return the canonical lowercase name for a benchmark scenario."""

    if not isinstance(value, str):
        raise ValueError(f"Unknown scenario: {value!r}")
    normalized = value.strip().casefold()
    try:
        return _SCENARIO_ALIASES[normalized]
    except KeyError as error:
        raise ValueError(f"Unknown scenario: {value!r}") from error


def normalize_options(value: dict | list) -> list[str]:
    """Return ordered, letter-prefixed option strings."""

    if isinstance(value, dict):
        entries = list(value.items())
    elif isinstance(value, list):
        entries = [
            (chr(ord("A") + index), option)
            for index, option in enumerate(value)
        ]
    else:
        raise ValueError("Options must be a dictionary or list.")

    options = []
    for label, option in entries:
        option_text = str(option).strip()
        if _OPTION_PREFIX.match(option_text):
            options.append(option_text)
            continue
        label_text = str(label).strip().rstrip(".):")
        options.append(f"{label_text}. {option_text}")
    return options


def build_tasks(record: dict) -> dict[str, dict[str, str]]:
    """Build the ten benchmark output tasks from an input record."""

    tasks = {
        "Sequential-QA": {
            "q": record["Sequential-QA"]["Question"],
            "a": record["Sequential-QA"]["Answer"],
        },
        "Prediction": {
            "q": record["Prediction"]["Question"],
            "a": record["Prediction"]["Answer"],
        },
        "Reasoning": _multiple_choice_task(record["Reasoning"]),
        "Description1": {
            "q": _DESCRIPTION_QUESTION,
            "a": record["Description1"],
        },
        "Description2": {
            "q": _DESCRIPTION_QUESTION,
            "a": record["Description2"],
        },
        "Caption": {
            "q": _CAPTION_QUESTION,
            "a": record["Caption"],
        },
        "static QA": {
            "q": record["static QA"]["Question"],
            "a": record["static QA"]["Answer"],
        },
        "MCQA": _multiple_choice_task(record["MCQA"]),
    }

    conversation = record["Conversation"]
    user_one = conversation["User 1"]
    assistant_one = conversation["Assistant 1"]
    tasks["Conversation1"] = {
        "q": f"User: {user_one} Assistant:",
        "a": assistant_one,
    }
    tasks["Conversation2"] = {
        "q": (
            f"User: {user_one} Assistant: {assistant_one} "
            f"User: {conversation['User 2']} Assistant:"
        ),
        "a": conversation["Assistant 2"],
    }
    return tasks


def build_prompt(scenario: str, task_name: str, question: str) -> str:
    """Build the full provider prompt for one benchmark task."""

    canonical = canonical_scenario(scenario)
    try:
        task_prompt = _TASK_PROMPTS[task_name]
    except KeyError as error:
        raise ValueError(f"Unknown task: {task_name!r}") from error

    if task_name in {"Conversation1", "Conversation2"}:
        question_prompt = question
    else:
        question_prompt = f"Here is the question: {question}"
    return " ".join(
        (_SCENARIO_PROMPTS[canonical], task_prompt, _NOTICE, question_prompt)
    )


def _multiple_choice_task(value: dict) -> dict[str, str]:
    question = " ".join(
        [value["Question"], *normalize_options(value["Options"])]
    )
    return {"q": question, "a": value["Correct Answer"]}

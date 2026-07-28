from __future__ import annotations

import pytest


@pytest.fixture
def benchmark_record():
    return {
        "id": "website-0",
        "video_path": "final_videos/website/0.mp4",
        "Sequential-QA": {
            "Question": "What did the user do before opening the menu?",
            "Answer": "The user selected the document.",
        },
        "Prediction": {
            "Question": "What will the user likely do next?",
            "Answer": "The user will choose an item from the menu.",
        },
        "Reasoning": {
            "Question": "Why did the menu appear?",
            "Options": [
                "A. The user clicked the menu button",
                "B. The page reloaded",
                "C. The app started",
                "D. The user closed the document",
            ],
            "Correct Answer": "A. The user clicked the menu button",
        },
        "Description1": "The user selects a document and opens a menu.",
        "Description2": "A document becomes selected before a menu appears.",
        "Caption": "Opening a menu for a selected document",
        "static QA": {
            "Question": "What color is the menu button?",
            "Answer": "The menu button is blue.",
        },
        "MCQA": {
            "Question": "Which element did the user select?",
            "Options": {
                "A": "A folder",
                "B": "A document",
                "C": "A toolbar",
                "D": "A dialog",
            },
            "Correct Answer": "B. A document",
        },
        "Conversation": {
            "User 1": "Which item is selected?",
            "Assistant 1": "The document is selected.",
            "User 2": "What opens next?",
            "Assistant 2": "A menu opens next.",
        },
    }

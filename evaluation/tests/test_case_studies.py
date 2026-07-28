import json
from pathlib import Path

import cv2
import numpy as np


SCENARIOS = {"android", "ios", "multi", "software", "website", "xr"}


def grayscale_correlation(left, right):
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY).astype(np.float64)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY).astype(np.float64)
    left_normalized = left_gray - left_gray.mean()
    right_normalized = right_gray - right_gray.mean()
    denominator = np.linalg.norm(left_normalized) * np.linalg.norm(
        right_normalized
    )
    assert denominator > 0
    return (
        np.dot(left_normalized.ravel(), right_normalized.ravel()) / denominator
    )


def test_all_case_studies_are_complete():
    root = Path(__file__).parents[2] / "case_studies"
    assert {path.name for path in root.iterdir() if path.is_dir()} == SCENARIOS

    for scenario in SCENARIOS:
        case_dir = root / scenario
        annotation = json.loads(
            (case_dir / "annotation.json").read_text(encoding="utf-8")
        )
        assert annotation["video_path"] == "example.mp4"

        video_path = case_dir / "example.mp4"
        assert video_path.is_file()
        capture = cv2.VideoCapture(str(video_path))
        assert capture.isOpened()
        keyframes = {
            keyframe["frame"]: case_dir / f"{keyframe['frame']}.png"
            for keyframe in annotation["keyframes"]
        }
        assert all(image_path.is_file() for image_path in keyframes.values())
        decoded_keyframes = set()
        try:
            frame_number = 0
            while True:
                decoded, video_frame = capture.read()
                if not decoded:
                    break
                image_path = keyframes.get(frame_number)
                if image_path is None:
                    frame_number += 1
                    continue
                submitted_image = cv2.imread(str(image_path))
                assert submitted_image is not None
                assert video_frame.shape == submitted_image.shape
                assert grayscale_correlation(video_frame, submitted_image) >= 0.95
                decoded_keyframes.add(frame_number)
                frame_number += 1
        finally:
            capture.release()
        assert decoded_keyframes == set(keyframes)

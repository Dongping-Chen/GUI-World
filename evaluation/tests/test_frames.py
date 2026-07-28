from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from evaluation import frames
from evaluation.frames import (
    FrameExtractionError,
    annotated_indices,
    canonical_strategy,
    extract_frames,
    linspace_indices,
)


def _write_video(path: Path, values: list[int]) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10,
        (32, 24),
    )
    assert writer.isOpened()
    try:
        for value in values:
            frame = np.full((24, 32, 3), value, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    return path


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("linspace", "linspace"),
        ("LiNsPaCe", "linspace"),
        ("annotated", "annotated"),
        ("ANNOTATED", "annotated"),
        ("Random", "linspace"),
        ("random", "linspace"),
        ("Extracted", "annotated"),
        ("EXTRACTED", "annotated"),
    ],
)
def test_canonical_strategy_accepts_canonical_and_legacy_names(value, expected):
    assert canonical_strategy(value) == expected


def test_canonical_strategy_rejects_unknown_names():
    with pytest.raises(FrameExtractionError, match="Unknown frame strategy"):
        canonical_strategy("scene-change")


@pytest.mark.parametrize(
    ("frame_count", "count", "expected"),
    [
        (1, 10, [0]),
        (5, 10, [0, 1, 2, 3, 4]),
        (10, 10, list(range(10))),
        (20, 5, [0, 4, 9, 14, 19]),
    ],
)
def test_linspace_indices_are_unique_valid_and_include_endpoints(
    frame_count,
    count,
    expected,
):
    indices = linspace_indices(frame_count, count)

    assert indices == expected
    assert len(indices) == len(set(indices))
    assert all(0 <= index < frame_count for index in indices)
    assert indices[0] == 0
    assert indices[-1] == frame_count - 1


@pytest.mark.parametrize(
    ("frame_count", "count"),
    [(0, 10), (-1, 10), (5, 0), (5, -1)],
)
def test_linspace_indices_reject_invalid_counts(frame_count, count):
    with pytest.raises(FrameExtractionError):
        linspace_indices(frame_count, count)


def test_annotated_indices_preserve_order_deduplicate_and_clamp():
    assert annotated_indices(
        [
            {"frame": 4},
            {"frame": 1},
            {"frame": 12},
            {"frame": 1},
            {"frame": 99},
            {"frame": 3},
        ],
        frame_count=5,
    ) == [4, 1, 3]


def test_annotated_indices_cap_at_ten_after_deduplication():
    keyframes = [{"frame": 0}, {"frame": 0}]
    keyframes.extend({"frame": index} for index in range(1, 11))

    assert annotated_indices(keyframes, frame_count=20) == list(range(10))


@pytest.mark.parametrize("frame", [-1, 1.0, "1", True, None])
def test_annotated_indices_reject_negative_and_non_integer_frames(frame):
    with pytest.raises(FrameExtractionError, match="non-negative integer"):
        annotated_indices([{"frame": frame}], frame_count=5)


def test_annotated_indices_reject_missing_frame_value():
    with pytest.raises(FrameExtractionError, match="frame"):
        annotated_indices([{"sub_goal": "Open Settings"}], frame_count=5)


@pytest.mark.parametrize("keyframes", [None, (), []])
def test_extract_frames_requires_annotations_for_annotated_strategy(
    tmp_path,
    keyframes,
):
    video_path = _write_video(tmp_path / "video.avi", [20, 60])

    with pytest.raises(
        FrameExtractionError,
        match=r"Android.*linspace|linspace.*Android",
    ):
        extract_frames(video_path, "annotated", keyframes=keyframes)


def test_extract_frames_decodes_once_and_returns_requested_annotation_order(
    tmp_path,
    monkeypatch,
):
    video_path = _write_video(
        tmp_path / "video.avi",
        [20, 60, 100, 140, 180],
    )
    real_video_capture = cv2.VideoCapture
    captures = []

    class TrackingCapture:
        def __init__(self, path):
            self.capture = real_video_capture(path)
            self.released = False
            captures.append(self)

        def isOpened(self):
            return self.capture.isOpened()

        def get(self, property_id):
            return self.capture.get(property_id)

        def read(self):
            return self.capture.read()

        def set(self, property_id, value):
            raise AssertionError("Frame extraction must not seek randomly.")

        def release(self):
            self.released = True
            self.capture.release()

    monkeypatch.setattr(frames.cv2, "VideoCapture", TrackingCapture)

    images = extract_frames(
        video_path,
        "annotated",
        keyframes=[
            {"frame": 4},
            {"frame": 1},
            {"frame": 3},
            {"frame": 1},
        ],
    )

    assert len(captures) == 1
    assert captures[0].released
    assert all(isinstance(image, Image.Image) for image in images)
    assert all(image.mode == "RGB" for image in images)
    assert [round(float(np.asarray(image).mean())) for image in images] == pytest.approx(
        [180, 60, 140],
        abs=5,
    )


@pytest.mark.parametrize(
    ("values", "expected_count"),
    [
        ([80], 1),
        ([20, 40, 60, 80, 100], 5),
    ],
)
def test_extract_frames_handles_short_linspace_videos(
    tmp_path,
    values,
    expected_count,
):
    video_path = _write_video(tmp_path / "short.avi", values)

    images = extract_frames(video_path, "linspace")

    assert len(images) == expected_count
    assert all(image.mode == "RGB" for image in images)


def test_extract_frames_rejects_missing_video(tmp_path):
    with pytest.raises(FrameExtractionError, match="does not exist"):
        extract_frames(tmp_path / "missing.mp4", "linspace")


def test_extract_frames_rejects_unreadable_video_and_releases_capture(
    tmp_path,
    monkeypatch,
):
    video_path = tmp_path / "corrupt.mp4"
    video_path.write_bytes(b"not a video")
    real_video_capture = cv2.VideoCapture
    captures = []

    class TrackingCapture:
        def __init__(self, path):
            self.capture = real_video_capture(path)
            self.released = False
            captures.append(self)

        def isOpened(self):
            return self.capture.isOpened()

        def get(self, property_id):
            return self.capture.get(property_id)

        def read(self):
            return self.capture.read()

        def release(self):
            self.released = True
            self.capture.release()

    monkeypatch.setattr(frames.cv2, "VideoCapture", TrackingCapture)

    with pytest.raises(FrameExtractionError, match="open|read"):
        extract_frames(video_path, "linspace")

    assert len(captures) == 1
    assert captures[0].released


def test_extract_frames_rejects_zero_frame_video_and_releases_capture(
    tmp_path,
    monkeypatch,
):
    video_path = tmp_path / "empty.avi"
    video_path.touch()

    class ZeroFrameCapture:
        def __init__(self, path):
            self.released = False

        def isOpened(self):
            return True

        def get(self, property_id):
            assert property_id == cv2.CAP_PROP_FRAME_COUNT
            return 0

        def release(self):
            self.released = True

    capture = ZeroFrameCapture(video_path)
    monkeypatch.setattr(frames.cv2, "VideoCapture", lambda path: capture)

    with pytest.raises(FrameExtractionError, match="zero frames"):
        extract_frames(video_path, "linspace")

    assert capture.released

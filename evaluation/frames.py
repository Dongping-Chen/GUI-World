from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class FrameExtractionError(RuntimeError):
    """Raised when video frames cannot be selected or decoded."""


_STRATEGY_ALIASES = {
    "linspace": "linspace",
    "random": "linspace",
    "annotated": "annotated",
    "extracted": "annotated",
}


def canonical_strategy(value: str) -> str:
    """Return the canonical name for a frame-selection strategy."""

    if not isinstance(value, str):
        raise FrameExtractionError(f"Unknown frame strategy: {value!r}")
    normalized = value.strip().casefold()
    try:
        return _STRATEGY_ALIASES[normalized]
    except KeyError as error:
        raise FrameExtractionError(
            f"Unknown frame strategy: {value!r}"
        ) from error


def linspace_indices(frame_count: int, count: int = 10) -> list[int]:
    """Return unique, evenly spaced frame indices including both endpoints."""

    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count <= 0
    ):
        raise FrameExtractionError("Frame count must be a positive integer.")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise FrameExtractionError("Selection count must be a positive integer.")

    candidates = np.linspace(
        0,
        frame_count - 1,
        num=count,
        dtype=int,
    ).tolist()
    return _deduplicate(candidates)


def annotated_indices(
    keyframes: list[dict],
    frame_count: int,
) -> list[int]:
    """Return validated annotation frame indices in annotation order."""

    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count <= 0
    ):
        raise FrameExtractionError("Frame count must be a positive integer.")
    if not keyframes:
        raise FrameExtractionError(
            "Annotated keyframes are unavailable. Android records do not "
            "provide annotations; use the linspace strategy instead."
        )

    indices = []
    seen = set()
    for annotation in keyframes:
        if not isinstance(annotation, dict) or "frame" not in annotation:
            raise FrameExtractionError(
                "Each keyframe annotation must contain a frame value."
            )
        frame = annotation["frame"]
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            raise FrameExtractionError(
                "Annotated frame values must be non-negative integers."
            )
        index = min(frame, frame_count - 1)
        if index in seen:
            continue
        indices.append(index)
        seen.add(index)
        if len(indices) == 10:
            break
    return indices


def extract_frames(
    video_path: Path,
    strategy: str,
    keyframes=(),
) -> list[Image.Image]:
    """Decode and return selected video frames as RGB Pillow images.

    The video is decoded twice: first to count actually decodable frames, then
    to extract selections without random seeking. This favors correct frame
    indices over single-pass performance.
    """

    canonical = canonical_strategy(strategy)
    if canonical == "annotated" and not keyframes:
        raise FrameExtractionError(
            "Annotated keyframes are unavailable. Android records do not "
            "provide annotations; use the linspace strategy instead."
        )

    path = Path(video_path)
    if not path.is_file():
        raise FrameExtractionError(f"Video does not exist: {path}")

    count_capture = cv2.VideoCapture(str(path))
    try:
        if not count_capture.isOpened():
            raise FrameExtractionError(f"Could not open or read video: {path}")

        frame_count = 0
        while True:
            decoded, _ = count_capture.read()
            if not decoded:
                break
            frame_count += 1
    finally:
        count_capture.release()

    if frame_count <= 0:
        raise FrameExtractionError(f"Video contains zero frames: {path}")

    if canonical == "linspace":
        indices = linspace_indices(frame_count)
    else:
        indices = annotated_indices(list(keyframes), frame_count)

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise FrameExtractionError(f"Could not reopen video: {path}")
        selected = {}
        targets = set(indices)
        last_index = max(indices)
        frame_index = 0
        while frame_index <= last_index:
            decoded, frame = capture.read()
            if not decoded:
                raise FrameExtractionError(
                    f"Could not read frame {frame_index} from video: {path}"
                )
            if frame_index in targets:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                selected[frame_index] = Image.fromarray(rgb_frame)
            frame_index += 1

        return [selected[index] for index in indices]
    finally:
        capture.release()


def _deduplicate(values: list[int]) -> list[int]:
    return list(dict.fromkeys(values))

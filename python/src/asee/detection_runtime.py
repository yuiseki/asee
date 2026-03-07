"""Detection pipeline helpers for YuNet-style face detectors."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

from .owner_policy import keep_largest_owner
from .tracking import FaceBox


class FrameLike(Protocol):
    """Minimal frame interface needed for detection-time geometry."""

    shape: Sequence[int]


class YunetDetectorLike(Protocol):
    """Minimal detector surface used by the extracted detection pipeline."""

    def detect(
        self,
        frame: FrameLike,
    ) -> tuple[object | None, Sequence[Sequence[float]] | None]: ...


type ClassifyLabelFn = Callable[[FrameLike, FaceBox], tuple[str, float]]


def to_square(
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    frame_w: int,
    frame_h: int,
    padding: float = 1.8,
) -> tuple[int, int, int, int]:
    """Expand a detected box to a padded square while staying inside the frame."""
    side = int(max(w, h) * padding)
    center_x = x + w // 2
    center_y = y + h // 2
    square_x = max(0, center_x - side // 2)
    square_y = max(0, center_y - side // 2)
    if square_x + side > frame_w:
        square_x = frame_w - side
    if square_y + side > frame_h:
        square_y = frame_h - side
    square_x = max(0, square_x)
    square_y = max(0, square_y)
    side = min(side, frame_w - square_x, frame_h - square_y)
    return square_x, square_y, side, side


class YunetDetectionPipeline:
    """Normalize YuNet detector output into FaceBox instances."""

    def __init__(
        self,
        *,
        detector: YunetDetectorLike,
        classify_label: ClassifyLabelFn,
        min_face_size: int = 20,
    ) -> None:
        self._detector = detector
        self._classify_label = classify_label
        self._min_face_size = min_face_size

    def detect_faces(self, frame: FrameLike) -> list[FaceBox]:
        frame_h, frame_w = int(frame.shape[0]), int(frame.shape[1])
        set_detector_input_size(self._detector, (frame_w, frame_h))
        _, detections = self._detector.detect(frame)
        if detections is None or len(detections) == 0:
            return []

        result: list[FaceBox] = []
        for detection in detections:
            x = max(0, int(detection[0]))
            y = max(0, int(detection[1]))
            detected_w = min(int(detection[2]), frame_w - x)
            detected_h = min(int(detection[3]), frame_h - y)
            confidence = float(detection[14])
            if detected_w <= 0 or detected_h <= 0:
                continue
            if detected_w < self._min_face_size or detected_h < self._min_face_size:
                continue
            square_x, square_y, square_w, square_h = to_square(
                x,
                y,
                detected_w,
                detected_h,
                frame_w=frame_w,
                frame_h=frame_h,
            )
            raw_detection = _copy_detection(detection)
            face_box = FaceBox(
                x=square_x,
                y=square_y,
                w=square_w,
                h=square_h,
                confidence=confidence,
                raw_detection=raw_detection,
            )
            face_box.label, face_box.confidence = self._classify_label(frame, face_box)
            result.append(face_box)

        return keep_largest_owner(result)


def _copy_detection(detection: Sequence[float]) -> Any:
    if hasattr(detection, "copy"):
        return detection.copy()
    return list(detection)


def set_detector_input_size(detector: object, size: tuple[int, int]) -> None:
    setter = getattr(detector, "setInputSize", None)
    if callable(setter):
        setter(size)
        return

    snake_case_setter = getattr(detector, "set_input_size", None)
    if callable(snake_case_setter):
        snake_case_setter(size)

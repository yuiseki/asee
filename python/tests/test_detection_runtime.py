"""Unit tests for the extracted YuNet detection pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from asee.detection_runtime import YunetDetectionPipeline, to_square


@dataclass
class FakeFrame:
    shape: tuple[int, int, int]


class FakeDetector:
    def __init__(self, detections: list[list[float]] | None) -> None:
        self.detections = detections
        self.input_sizes: list[tuple[int, int]] = []

    def set_input_size(self, size: tuple[int, int]) -> None:
        self.input_sizes.append(size)

    def detect(self, frame: FakeFrame) -> tuple[None, list[list[float]] | None]:
        del frame
        return None, self.detections


def make_detection(x: int, y: int, w: int, h: int, conf: float = 0.9) -> list[float]:
    row = [0.0] * 15
    row[0], row[1], row[2], row[3], row[14] = float(x), float(y), float(w), float(h), conf
    return row


def test_to_square_keeps_box_inside_frame():
    assert to_square(100, 120, 80, 120, frame_w=320, frame_h=240) == (32, 24, 216, 216)


def test_detect_faces_filters_tiny_faces():
    detector = FakeDetector([make_detection(100, 100, 10, 10)])
    pipeline = YunetDetectionPipeline(
        detector=detector,
        classify_label=lambda _frame, _fb: ('SUBJECT', 0.0),
    )

    result = pipeline.detect_faces(FakeFrame(shape=(720, 1280, 3)))

    assert result == []


def test_detect_faces_keeps_face_at_min_size():
    detector = FakeDetector([make_detection(100, 100, 20, 20)])
    pipeline = YunetDetectionPipeline(
        detector=detector,
        classify_label=lambda _frame, _fb: ('OWNER', 0.7),
    )

    result = pipeline.detect_faces(FakeFrame(shape=(720, 1280, 3)))

    assert len(result) == 1
    assert result[0].label == 'OWNER'


def test_detect_faces_sets_detector_input_size_and_passes_square_box_to_classifier():
    detector = FakeDetector([make_detection(100, 100, 100, 80)])
    observed: list[tuple[int, int, int, int]] = []

    def classify(_frame: FakeFrame, face_box) -> tuple[str, float]:
        observed.append((face_box.x, face_box.y, face_box.w, face_box.h))
        return 'SUBJECT', 0.8

    pipeline = YunetDetectionPipeline(detector=detector, classify_label=classify)

    result = pipeline.detect_faces(FakeFrame(shape=(720, 1280, 3)))

    assert detector.input_sizes == [(1280, 720)]
    assert len(result) == 1
    assert observed == [(60, 50, 180, 180)]


def test_detect_faces_keeps_only_largest_owner():
    detector = FakeDetector(
        [
            make_detection(10, 10, 80, 80),
            make_detection(200, 200, 120, 120),
            make_detection(600, 100, 90, 90),
        ]
    )
    labels = iter([('OWNER', 0.7), ('OWNER', 0.72), ('SUBJECT', 0.5)])
    pipeline = YunetDetectionPipeline(
        detector=detector,
        classify_label=lambda _frame, _fb: next(labels),
    )

    result = pipeline.detect_faces(FakeFrame(shape=(720, 1280, 3)))

    owners = [face for face in result if face.label == 'OWNER']
    subjects = [face for face in result if face.label == 'SUBJECT']
    assert len(owners) == 1
    assert owners[0].w >= 120
    assert len(subjects) == 1

"""Unit tests for the extracted GOD MODE overlay runtime."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from asee.overlay import GodModeOverlay
from asee.tracking import FaceBox


@pytest.fixture
def overlay() -> GodModeOverlay:
    return GodModeOverlay(width=1280, height=720)


@pytest.fixture
def blank_frame() -> np.ndarray:
    return np.zeros((720, 1280, 3), dtype=np.uint8)


def make_fake_detection(x: int, y: int, w: int, h: int, conf: float = 0.9) -> np.ndarray:
    row = np.zeros(15, dtype=np.float32)
    row[0], row[1], row[2], row[3], row[14] = x, y, w, h, conf
    return row


def test_draw_returns_numpy_array(overlay: GodModeOverlay, blank_frame: np.ndarray):
    result = overlay.draw(blank_frame, frame_count=0)

    assert isinstance(result, np.ndarray)
    assert result.shape == (720, 1280, 3)


def test_draw_modifies_frame(overlay: GodModeOverlay, blank_frame: np.ndarray):
    result = overlay.draw(blank_frame.copy(), frame_count=0)

    assert not np.array_equal(result, blank_frame)


def test_setters_update_runtime_state(overlay: GodModeOverlay):
    overlay.set_caption('室内。デスクに着席中。')
    overlay.set_prediction('次の20分: 飲料摂取 72%')

    assert overlay.caption == '室内。デスクに着席中。'
    assert overlay.prediction == '次の20分: 飲料摂取 72%'


def test_draw_with_owner_face_box_differs_from_subject(
    overlay: GodModeOverlay,
    blank_frame: np.ndarray,
):
    owner = overlay.draw(
        blank_frame.copy(),
        frame_count=0,
        face_boxes=[FaceBox(x=100, y=80, w=120, h=140, label='OWNER')],
        smooth=False,
    )
    subject = overlay.draw(
        blank_frame.copy(),
        frame_count=0,
        face_boxes=[FaceBox(x=100, y=80, w=120, h=140, label='SUBJECT')],
        smooth=False,
    )

    assert not np.array_equal(owner, subject)


def test_detect_faces_returns_empty_list_when_no_detectors(
    overlay: GodModeOverlay,
    blank_frame: np.ndarray,
):
    overlay._yunet_pipeline = None
    overlay._haar = None

    assert overlay.detect_faces(blank_frame) == []


def test_owner_embedding_is_none_by_default(overlay: GodModeOverlay):
    assert overlay._owner_embeddings is None


def test_set_owner_embedding_accepts_1d_and_2d_arrays(overlay: GodModeOverlay):
    overlay.set_owner_embedding(np.zeros(128, dtype=np.float32))
    assert overlay._owner_embeddings is not None
    assert overlay._owner_embeddings.shape == (1, 128)

    overlay.set_owner_embedding(np.zeros((2, 128), dtype=np.float32))
    assert overlay._owner_embeddings.shape == (2, 128)


def test_detect_yunet_filters_tiny_faces(overlay: GodModeOverlay, blank_frame: np.ndarray):
    overlay._detector = MagicMock()
    overlay._detector.detect.return_value = (
        None,
        np.array([make_fake_detection(100, 100, 10, 10)]),
    )
    overlay._yunet_pipeline = None

    with patch.object(overlay, '_classify_label', return_value=('SUBJECT', 0.0)):
        result = overlay._detect_yunet(blank_frame)

    assert result == []


def test_detect_yunet_keeps_face_at_min_size(overlay: GodModeOverlay, blank_frame: np.ndarray):
    overlay._detector = MagicMock()
    overlay._detector.detect.return_value = (
        None,
        np.array([make_fake_detection(100, 100, 20, 20)]),
    )
    overlay._yunet_pipeline = None

    with patch.object(overlay, '_classify_label', return_value=('OWNER', 0.7)):
        result = overlay._detect_yunet(blank_frame)

    assert len(result) == 1
    assert result[0].label == 'OWNER'


def test_detect_faces_keeps_only_largest_owner(overlay: GodModeOverlay, blank_frame: np.ndarray):
    overlay._detector = MagicMock()
    overlay._detector.detect.return_value = (
        None,
        np.array(
            [
                make_fake_detection(10, 10, 80, 80),
                make_fake_detection(200, 200, 120, 120),
                make_fake_detection(600, 100, 90, 90),
            ]
        ),
    )
    overlay._yunet_pipeline = None
    labels = iter([('OWNER', 0.7), ('OWNER', 0.72), ('SUBJECT', 0.5)])

    with patch.object(overlay, '_classify_label', side_effect=lambda _frame, _fb: next(labels)):
        result = overlay.detect_faces(blank_frame)

    owners = [face for face in result if face.label == 'OWNER']
    subjects = [face for face in result if face.label == 'SUBJECT']
    assert len(owners) == 1
    assert owners[0].w >= 120
    assert len(subjects) == 1


def test_face_capture_writer_enabled_with_dir(tmp_path):
    overlay = GodModeOverlay(width=320, height=240, face_capture_dir=str(tmp_path))

    assert overlay._face_capture_writer is not None
    assert overlay._face_capture_writer._min_interval == 10.0
    assert overlay._face_capture_writer._max_per_day == 500_000
    assert overlay._face_capture_writer._max_total == 500_000
    assert overlay._face_capture_writer._max_bytes == 51_200 * 1024 * 1024


def test_subject_capture_writer_uses_guest_collection_defaults(tmp_path):
    overlay = GodModeOverlay(width=320, height=240, subject_capture_dir=str(tmp_path))

    assert overlay._subject_capture_writer is not None
    assert overlay._subject_capture_writer._min_interval == 10.0
    assert overlay._subject_capture_writer._max_per_day == 500_000
    assert overlay._subject_capture_writer._max_total == 500_000
    assert overlay._subject_capture_writer._max_bytes == 51_200 * 1024 * 1024


def test_classify_label_saves_subject_crop(blank_frame: np.ndarray):
    overlay = GodModeOverlay(width=320, height=240, subject_capture_dir="/tmp/subject-capture-test")
    writer = MagicMock()
    overlay._subject_capture_writer = writer
    overlay._face_capture_writer = None
    overlay._owner_embeddings = None
    face = FaceBox(x=10, y=20, w=30, h=40)

    label, score = overlay._classify_label(blank_frame.copy(), face)

    assert label == "SUBJECT"
    writer.save.assert_called_once()
    crop_arg, score_arg = writer.save.call_args.args
    assert crop_arg.shape == (40, 30, 3)
    assert score_arg == score


def test_classify_label_saves_owner_crop(blank_frame: np.ndarray):
    overlay = GodModeOverlay(width=320, height=240, face_capture_dir="/tmp/owner-capture-test")
    writer = MagicMock()
    overlay._face_capture_writer = writer
    overlay._subject_capture_writer = None
    overlay._owner_embeddings = np.zeros((1, 128), dtype=np.float32)
    overlay._recognizer = MagicMock()
    overlay._recognizer.match.return_value = 0.9
    with patch.object(overlay, "extract_embedding", return_value=np.zeros((1, 128), dtype=np.float32)):
        label, score = overlay._classify_label(
            blank_frame.copy(),
            FaceBox(x=15, y=25, w=35, h=45),
        )

    assert label == "OWNER"
    writer.save.assert_called_once()
    crop_arg, score_arg = writer.save.call_args.args
    assert crop_arg.shape == (45, 35, 3)
    assert score_arg == score


def test_detection_backend_opencv_default():
    """Default backend must load cv2.FaceDetectorYN (or None if model missing)."""
    overlay = GodModeOverlay(width=320, height=240)
    # detector is either a cv2.FaceDetectorYN or None — never a GpuYuNetDetector
    from asee.gpu_yunet import GpuYuNetDetector

    assert not isinstance(overlay._detector, GpuYuNetDetector)


def test_detection_backend_onnxruntime():
    """onnxruntime backend must create a GpuYuNetDetector and GpuSFaceRecognizer."""
    from asee.gpu_sface import GpuSFaceRecognizer
    from asee.gpu_yunet import GpuYuNetDetector

    overlay = GodModeOverlay(
        width=640,
        height=640,
        detection_backend="onnxruntime",
    )
    assert isinstance(overlay._detector, GpuYuNetDetector)
    assert isinstance(overlay._recognizer, GpuSFaceRecognizer)


def test_detection_backend_onnxruntime_detects_no_faces_on_blank():
    """onnxruntime backend detect_faces on blank frame must return empty list."""
    overlay = GodModeOverlay(width=640, height=640, detection_backend="onnxruntime")
    blank = np.zeros((640, 640, 3), dtype=np.uint8)
    faces = overlay.detect_faces(blank)
    assert isinstance(faces, list)
    assert len(faces) == 0

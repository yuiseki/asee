"""Unit tests for extracted face tracking primitives."""

from __future__ import annotations

from asee.tracking import FaceBox, FaceTracker


def test_face_box_defaults_to_subject():
    face = FaceBox(x=0, y=0, w=50, h=50)

    assert face.label == 'SUBJECT'
    assert face.confidence == 1.0
    assert face.id == 0


def test_face_box_corners():
    face = FaceBox(x=10, y=20, w=100, h=80)

    assert face.corners() == (
      (10, 20),
      (110, 20),
      (10, 100),
      (110, 100),
    )


def test_face_box_iou():
    left = FaceBox(x=10, y=10, w=100, h=100)
    right = FaceBox(x=60, y=60, w=100, h=100)

    assert round(left.iou(right), 3) == 0.143


def test_face_tracker_hides_new_tracks_until_min_hits():
    tracker = FaceTracker(alpha=0.4, max_lost_frames=2, min_hits=2)

    first = tracker.update([FaceBox(x=10, y=10, w=80, h=80)])
    second = tracker.update([FaceBox(x=12, y=12, w=80, h=80)])

    assert first == []
    assert len(second) == 1
    assert second[0].id == 1


def test_face_tracker_assigns_new_ids_for_new_faces():
    tracker = FaceTracker(alpha=0.4, max_lost_frames=2, min_hits=1)

    tracked = tracker.update(
        [
            FaceBox(x=10, y=10, w=80, h=80),
            FaceBox(x=200, y=200, w=90, h=90),
        ]
    )

    assert [face.id for face in tracked] == [1, 2]


def test_face_tracker_keeps_recent_track_for_one_lost_frame():
    tracker = FaceTracker(alpha=0.4, max_lost_frames=2, min_hits=1)
    tracker.update([FaceBox(x=10, y=10, w=80, h=80)])

    tracked = tracker.update([])

    assert len(tracked) == 1
    assert tracked[0].id == 1


def test_face_tracker_drops_track_after_max_lost_frames():
    tracker = FaceTracker(alpha=0.4, max_lost_frames=2, min_hits=1)
    tracker.update([FaceBox(x=10, y=10, w=80, h=80)])

    tracker.update([])
    tracked = tracker.update([])

    assert tracked == []


# ---------------------------------------------------------------------------
# FaceBox.from_yunet_row
# ---------------------------------------------------------------------------

def _make_yunet_row(x=10.0, y=20.0, w=100.0, h=80.0, confidence=0.9):
    """Build a 15-element YuNet detection row: [x, y, w, h, kps×10, score]."""
    import numpy as np
    row = np.zeros(15, dtype=np.float32)
    row[0] = x
    row[1] = y
    row[2] = w
    row[3] = h
    row[14] = confidence
    return row


def test_from_yunet_row_creates_face_box():
    row = _make_yunet_row()
    face = FaceBox.from_yunet_row(row)
    assert isinstance(face, FaceBox)


def test_from_yunet_row_xywh():
    row = _make_yunet_row(x=10.0, y=20.0, w=100.0, h=80.0)
    face = FaceBox.from_yunet_row(row)
    assert face.x == 10
    assert face.y == 20
    assert face.w == 100
    assert face.h == 80


def test_from_yunet_row_confidence():
    row = _make_yunet_row(confidence=0.85)
    face = FaceBox.from_yunet_row(row)
    assert abs(face.confidence - 0.85) < 1e-5


def test_from_yunet_row_preserves_raw_detection():
    row = _make_yunet_row()
    face = FaceBox.from_yunet_row(row)
    assert face.raw_detection is row


def test_from_yunet_row_default_label_is_subject():
    face = FaceBox.from_yunet_row(_make_yunet_row())
    assert face.label == "SUBJECT"

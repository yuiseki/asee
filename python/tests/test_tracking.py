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

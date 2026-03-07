from types import SimpleNamespace

from asee.biometric_status import BiometricStatusTracker


def test_biometric_status_defaults_when_no_owner_seen() -> None:
    tracker = BiometricStatusTracker()

    status = tracker.snapshot(running=False, owner_embedding_loaded=False, now=100.0)

    assert status["running"] is False
    assert status["ownerEmbeddingLoaded"] is False
    assert status["ownerPresent"] is False
    assert status["ownerCount"] == 0
    assert status["subjectCount"] == 0
    assert status["peopleCount"] == 0
    assert status["ownerSeenAgoMs"] is None
    assert status["updatedAt"] == 100.0


def test_biometric_status_reports_owner_presence_and_recent_seen() -> None:
    tracker = BiometricStatusTracker()
    tracker.record_faces(
        [
            SimpleNamespace(label="OWNER"),
            SimpleNamespace(label="SUBJECT"),
        ],
        seen_at=120.0,
    )

    status = tracker.snapshot(running=True, owner_embedding_loaded=True, now=120.25)

    assert status["running"] is True
    assert status["ownerEmbeddingLoaded"] is True
    assert status["ownerPresent"] is True
    assert status["ownerCount"] == 1
    assert status["subjectCount"] == 1
    assert status["peopleCount"] == 2
    assert status["ownerSeenAgoMs"] == 250


def test_biometric_status_aggregates_faces_across_multiple_cameras() -> None:
    tracker = BiometricStatusTracker(camera_ids=[0, 2])
    tracker.record_faces([SimpleNamespace(label="OWNER")], camera_id=0, seen_at=50.0)
    tracker.record_faces(
        [SimpleNamespace(label="SUBJECT"), SimpleNamespace(label="GUEST")],
        camera_id=2,
        seen_at=50.1,
    )

    status = tracker.snapshot(running=True, owner_embedding_loaded=False, now=50.2)

    assert status["ownerPresent"] is True
    assert status["ownerCount"] == 1
    assert status["subjectCount"] == 2
    assert status["peopleCount"] == 3
    assert status["ownerSeenAgoMs"] == 200


def test_biometric_status_replaces_faces_for_a_camera_on_each_update() -> None:
    tracker = BiometricStatusTracker(camera_ids=[0])
    tracker.record_faces(
        [SimpleNamespace(label="OWNER"), SimpleNamespace(label="SUBJECT")],
        camera_id=0,
        seen_at=10.0,
    )
    tracker.record_faces([SimpleNamespace(label="SUBJECT")], camera_id=0, seen_at=10.5)

    status = tracker.snapshot(running=True, owner_embedding_loaded=True, now=11.0)

    assert status["ownerPresent"] is False
    assert status["ownerCount"] == 0
    assert status["subjectCount"] == 1
    assert status["peopleCount"] == 1
    assert status["ownerSeenAgoMs"] == 1000

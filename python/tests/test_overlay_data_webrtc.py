"""Tests for WebRTC overlay JSON payloads."""

from __future__ import annotations

from asee.overlay_data import FaceDetection, OverlayFrame


def test_overlay_frame_round_trips_via_json() -> None:
    frame = OverlayFrame(
        seq=42,
        ts_ms=123456,
        camera_id=6,
        caption="OBSERVING",
        prediction="OWNER PRESENT",
        faces=[
            FaceDetection(
                x=10,
                y=20,
                w=30,
                h=40,
                label="OWNER",
                confidence=0.98,
            )
        ],
    )

    restored = OverlayFrame.from_json(frame.to_json())

    assert restored == frame


def test_face_detection_to_dict_is_json_ready() -> None:
    detection = FaceDetection(
        x=1,
        y=2,
        w=3,
        h=4,
        label="SUBJECT",
        confidence=0.5,
    )

    assert detection.to_dict() == {
        "x": 1,
        "y": 2,
        "w": 3,
        "h": 4,
        "label": "SUBJECT",
        "confidence": 0.5,
    }

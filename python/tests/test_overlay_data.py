"""Tests for WebRTC overlay payload helpers."""

from __future__ import annotations

from asee.overlay_data import FaceDetection, OverlayFrame


def test_overlay_frame_json_roundtrip_preserves_frame_dimensions() -> None:
    payload = OverlayFrame(
        seq=1,
        ts_ms=2,
        camera_id=4,
        frame_width=1280,
        frame_height=720,
        caption="OBSERVING",
        prediction="OWNER PRESENT",
        faces=[
            FaceDetection(x=10, y=20, w=30, h=40, label="OWNER", confidence=0.98),
        ],
    )

    decoded = OverlayFrame.from_json(payload.to_json())

    assert decoded.camera_id == 4
    assert decoded.frame_width == 1280
    assert decoded.frame_height == 720
    assert decoded.faces[0].label == "OWNER"

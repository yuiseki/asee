"""Unit tests for the extracted server runtime state."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from asee.server_runtime import SeeingServerRuntime


class FakeOverlay:
    """Small overlay stub so runtime tests stay independent from OpenCV state."""

    def __init__(self) -> None:
        self.caption = ""
        self.prediction = ""
        self._owner_embeddings: np.ndarray | None = None

    def set_caption(self, text: str) -> None:
        self.caption = text

    def set_prediction(self, text: str) -> None:
        self.prediction = text

    def set_owner_embedding(self, embedding: np.ndarray) -> None:
        self._owner_embeddings = embedding


def make_frame(width: int = 1280, height: int = 720) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_update_overlay_text_updates_overlay_state_and_overlay():
    overlay = FakeOverlay()
    runtime = SeeingServerRuntime(overlay=overlay)

    runtime.update_overlay_text(caption="観測中", prediction="次の5分: 着席継続")

    assert runtime.overlay_state.caption == "観測中"
    assert runtime.overlay_state.prediction == "次の5分: 着席継続"
    assert overlay.caption == "観測中"
    assert overlay.prediction == "次の5分: 着席継続"


def test_get_snapshot_jpeg_returns_none_when_no_frame():
    runtime = SeeingServerRuntime(overlay=FakeOverlay())

    assert runtime.get_snapshot_jpeg() is None


def test_get_snapshot_jpeg_uses_encoder_and_quality():
    calls: list[tuple[np.ndarray, int]] = []

    def fake_encoder(frame: np.ndarray, quality: int) -> bytes:
        calls.append((frame, quality))
        return b"jpeg"

    runtime = SeeingServerRuntime(
        overlay=FakeOverlay(),
        jpeg_encoder=fake_encoder,
        jpeg_quality=87,
    )
    frame = make_frame()
    runtime.update_frame(frame)

    assert runtime.get_snapshot_jpeg() == b"jpeg"
    assert calls == [(frame, 87)]


def test_primary_camera_frame_becomes_snapshot_source():
    runtime = SeeingServerRuntime(
        overlay=FakeOverlay(),
        camera_ids=(0, 2),
        jpeg_encoder=lambda frame, quality: bytes([frame.shape[1] // 10, quality]),
        jpeg_quality=80,
    )

    runtime.update_frame(make_frame(width=640, height=480), camera_id=2)
    assert runtime.get_snapshot_jpeg() is None

    runtime.update_frame(make_frame(width=1280, height=720), camera_id=0)

    assert runtime.get_snapshot_jpeg() == bytes([128, 80])


def test_record_faces_reports_owner_presence_and_recent_seen():
    runtime = SeeingServerRuntime(overlay=FakeOverlay())
    runtime.set_running(True)

    runtime.record_faces(
        [
            SimpleNamespace(label="OWNER"),
            SimpleNamespace(label="SUBJECT"),
        ],
        seen_at=100.0,
    )
    status = runtime.get_biometric_status(now=100.25)

    assert status["running"] is True
    assert status["ownerPresent"] is True
    assert status["ownerCount"] == 1
    assert status["subjectCount"] == 1
    assert status["peopleCount"] == 2
    assert status["ownerSeenAgoMs"] == 250


def test_multicamera_runtime_requires_camera_id_for_faces():
    runtime = SeeingServerRuntime(overlay=FakeOverlay(), camera_ids=(0, 2))

    with pytest.raises(ValueError, match="camera_id is required"):
        runtime.record_faces([SimpleNamespace(label="OWNER")])


def test_load_owner_embedding_from_npy_file(tmp_path):
    overlay = FakeOverlay()
    runtime = SeeingServerRuntime(overlay=overlay)
    path = tmp_path / "owner.npy"
    expected = np.ones((2, 128), dtype=np.float32)
    np.save(path, expected)

    runtime.load_owner_embedding(path)

    assert runtime.owner_embedding_loaded is True
    assert overlay._owner_embeddings is not None
    assert np.array_equal(overlay._owner_embeddings, expected)


def test_iter_mjpeg_delegates_to_stream_factory():
    calls: list[int | None] = []

    def fake_stream_factory(device: int | None) -> Any:
        calls.append(device)
        return [b"a", b"b"]

    runtime = SeeingServerRuntime(
        overlay=FakeOverlay(),
        stream_factory=fake_stream_factory,
    )

    assert list(runtime.iter_mjpeg()) == [b"a", b"b"]
    assert list(runtime.iter_mjpeg(4)) == [b"a", b"b"]
    assert calls == [None, 4]


def test_transport_defaults_to_webrtc_and_is_mutable():
    runtime = SeeingServerRuntime(overlay=FakeOverlay())

    assert runtime.transport == "webrtc"

    runtime.transport = "mjpeg"

    assert runtime.transport == "mjpeg"


def test_frame_revision_increments_when_primary_frame_updates():
    runtime = SeeingServerRuntime(overlay=FakeOverlay())

    first = runtime.get_frame_revision()
    runtime.update_frame(make_frame())
    second = runtime.get_frame_revision()

    assert first == 0
    assert second == 1


def test_wait_for_frame_update_returns_new_revision_before_timeout():
    runtime = SeeingServerRuntime(overlay=FakeOverlay(), camera_ids=(2,))
    observed: list[int] = []

    def wait_in_thread() -> None:
        observed.append(
            runtime.wait_for_frame_update(
                camera_id=2,
                after_revision=0,
                timeout_sec=0.5,
            )
        )

    worker = threading.Thread(target=wait_in_thread)
    worker.start()
    time.sleep(0.05)
    runtime.update_frame(make_frame(width=640, height=480), camera_id=2)
    worker.join(timeout=1.0)

    assert observed == [1]

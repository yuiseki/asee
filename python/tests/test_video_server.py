"""Unit tests for the extracted GOD MODE video server."""

from __future__ import annotations

import json
import threading
import time
from urllib.request import Request, urlopen

import numpy as np

from asee.tracking import FaceBox
from asee.video_server import GodModeVideoServer, encode_frame_to_jpeg


def wait_until(predicate: object, *, timeout: float = 2.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if callable(predicate) and predicate():
            return True
        time.sleep(0.05)
    return False


class TestEncodeFrameToJpeg:
    def test_returns_bytes(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = encode_frame_to_jpeg(frame)

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_jpeg_header(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = encode_frame_to_jpeg(frame)

        assert result[0] == 0xFF
        assert result[1] == 0xD8

    def test_quality_affects_size(self) -> None:
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        small = encode_frame_to_jpeg(frame, quality=30)
        large = encode_frame_to_jpeg(frame, quality=90)

        assert len(small) < len(large)


class TestGodModeVideoServer:
    def test_instantiation(self) -> None:
        server = GodModeVideoServer(port=18865, device_index=0)

        assert server.port == 18865
        assert server.is_running is False

    def test_update_frame(self) -> None:
        server = GodModeVideoServer(port=18866, device_index=0)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        server.update_frame(frame)

        assert server.current_frame is not None

    def test_update_overlay_text(self) -> None:
        server = GodModeVideoServer(port=18867, device_index=0)

        server.update_overlay_text(caption="テスト観測", prediction="テスト予測")

        assert server.overlay.caption == "テスト観測"
        assert server.overlay.prediction == "テスト予測"
        assert server.runtime.overlay_state.caption == "テスト観測"
        assert server.runtime.overlay_state.prediction == "テスト予測"

    def test_biometric_status_defaults_when_no_owner_seen(self) -> None:
        server = GodModeVideoServer(port=18868, device_index=None)

        status = server.get_biometric_status()

        assert status["running"] is False
        assert status["ownerPresent"] is False
        assert status["ownerCount"] == 0
        assert status["subjectCount"] == 0
        assert status["ownerSeenAgoMs"] is None

    def test_biometric_status_reports_owner_presence_and_recent_seen(self) -> None:
        server = GodModeVideoServer(port=18869, device_index=None)

        server._record_owner_presence(
            [
                FaceBox(x=0, y=0, w=10, h=10, label="OWNER"),
                FaceBox(x=20, y=20, w=10, h=10, label="SUBJECT"),
            ]
        )
        status = server.get_biometric_status()

        assert status["ownerPresent"] is True
        assert status["ownerCount"] == 1
        assert status["subjectCount"] == 1
        assert isinstance(status["ownerSeenAgoMs"], int)
        assert status["ownerSeenAgoMs"] >= 0

    def test_server_starts_and_stops(self) -> None:
        server = GodModeVideoServer(port=18870, device_index=None)
        thread = threading.Thread(target=server.start, daemon=True)

        thread.start()
        assert wait_until(lambda: server.is_running)

        server.stop()
        thread.join(timeout=3.0)

        assert server.is_running is False

    def test_http_root_returns_html(self) -> None:
        server = GodModeVideoServer(port=18871, device_index=None)
        thread = threading.Thread(target=server.start, daemon=True)
        thread.start()
        assert wait_until(lambda: server.is_running)

        try:
            with urlopen("http://127.0.0.1:18871/", timeout=3) as response:
                body = response.read().decode("utf-8")
                assert response.status == 200
                assert 'rel="manifest"' in body
                assert "serviceWorker.register" in body
        finally:
            server.stop()
            thread.join(timeout=3.0)

    def test_http_update_endpoint(self) -> None:
        server = GodModeVideoServer(port=18872, device_index=None)
        thread = threading.Thread(target=server.start, daemon=True)
        thread.start()
        assert wait_until(lambda: server.is_running)

        try:
            request = Request(
                "http://127.0.0.1:18872/update",
                data=json.dumps(
                    {"caption": "観測テスト", "prediction": "予測テスト"}
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=3) as response:
                payload = json.loads(response.read().decode("utf-8"))
                assert response.status == 200
                assert payload["status"] == "ok"
            assert server.overlay.caption == "観測テスト"
            assert server.overlay.prediction == "予測テスト"
        finally:
            server.stop()
            thread.join(timeout=3.0)

    def test_http_manifest_endpoint(self) -> None:
        server = GodModeVideoServer(
            port=18873,
            device_index=None,
            title="GOD MODE 18873",
        )
        thread = threading.Thread(target=server.start, daemon=True)
        thread.start()
        assert wait_until(lambda: server.is_running)

        try:
            with urlopen(
                "http://127.0.0.1:18873/manifest.webmanifest",
                timeout=3,
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
                assert response.status == 200
                assert payload["name"] == "GOD MODE 18873"
                assert payload["display"] == "standalone"
                assert payload["start_url"] == "/"
                assert payload["icons"]
        finally:
            server.stop()
            thread.join(timeout=3.0)


class TestOpenCameraResolution:
    def test_open_camera_sets_fourcc_mjpg(self) -> None:
        from unittest.mock import MagicMock, patch

        import cv2

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 1280.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 720.0,
        }.get(prop, 0.0)

        fourcc_calls: list[int] = []

        def cap_set(prop: int, value: float) -> bool:
            if prop == cv2.CAP_PROP_FOURCC:
                fourcc_calls.append(int(value))
            return True

        mock_cap.set.side_effect = cap_set

        with patch("cv2.VideoCapture", return_value=mock_cap):
            server = GodModeVideoServer(device_index=None)
            server._open_camera(0)

        expected_mjpg = cv2.VideoWriter_fourcc(*"MJPG")
        assert expected_mjpg in fourcc_calls

    def test_captured_frame_resized_to_output_resolution(self) -> None:
        server = GodModeVideoServer(device_index=None, width=1280, height=720)
        small_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        normalized = server._normalize_frame(small_frame)

        assert normalized.shape == (720, 1280, 3)

    def test_normalize_frame_passthrough_when_already_correct_size(self) -> None:
        server = GodModeVideoServer(device_index=None, width=1280, height=720)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        normalized = server._normalize_frame(frame)

        assert normalized.shape == (720, 1280, 3)
        assert normalized is frame

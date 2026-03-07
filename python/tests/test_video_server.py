"""Unit tests for the extracted GOD MODE video server."""

from __future__ import annotations

import json
import threading
import time
from unittest.mock import call, patch
from urllib.request import Request, urlopen

import numpy as np
import pytest

from asee.tracking import FaceBox
from asee.video_server import (
    CaptureSettings,
    GodModeVideoServer,
    LiveCameraDisabledError,
    decode_fourcc_value,
    encode_frame_to_jpeg,
)


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
        server = GodModeVideoServer(port=18865, device_index=0, allow_live_camera=True)

        assert server.port == 18865
        assert server.is_running is False

    def test_single_camera_defaults_to_720p_30fps_mjpg(self) -> None:
        server = GodModeVideoServer(port=188651, device_index=0, allow_live_camera=True)

        assert server.capture_settings == CaptureSettings(
            width=1280,
            height=720,
            fps=30.0,
            fourcc="MJPG",
        )

    def test_multicamera_defaults_to_lower_risk_capture_profile(self) -> None:
        server = GodModeVideoServer(
            port=188652,
            device_index=None,
            camera_list=[0, 2, 4, 6],
            allow_live_camera=True,
        )

        assert server.capture_settings == CaptureSettings(
            width=640,
            height=360,
            fps=10.0,
            fourcc="MJPG",
        )

    def test_explicit_capture_profile_override_is_preserved(self) -> None:
        server = GodModeVideoServer(
            port=188653,
            device_index=None,
            camera_list=[0, 2],
            allow_live_camera=True,
            width=800,
            height=600,
            fps=12.5,
            fourcc="YUYV",
        )

        assert server.capture_settings == CaptureSettings(
            width=800,
            height=600,
            fps=12.5,
            fourcc="YUYV",
        )

    def test_update_frame(self) -> None:
        server = GodModeVideoServer(port=18866, device_index=None)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        server.update_frame(frame)

        assert server.current_frame is not None

    def test_update_overlay_text(self) -> None:
        server = GodModeVideoServer(port=18867, device_index=None)

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

    def test_live_camera_requires_explicit_opt_in_for_single_camera(self) -> None:
        with pytest.raises(LiveCameraDisabledError, match="allow-live-camera"):
            GodModeVideoServer(port=18869, device_index=0)

    def test_live_camera_requires_explicit_opt_in_for_multicamera(self) -> None:
        with pytest.raises(LiveCameraDisabledError, match="allow-live-camera"):
            GodModeVideoServer(port=18869, device_index=None, camera_list=[0, 2, 4, 6])

    def test_start_emits_diagnostics_events_and_memory_samples(self) -> None:
        events: list[tuple[str, dict[str, object]]] = []

        class FakeDiagnosticsLogger:
            @property
            def path(self) -> str:
                return "/tmp/fake-asee-diagnostics.jsonl"

            def log_event(self, event: str, **fields: object) -> None:
                events.append((event, dict(fields)))

            def open_fault_handler_stream(self) -> None:
                return None

            def close(self) -> None:
                return None

        server = GodModeVideoServer(
            port=18870,
            device_index=None,
            diagnostics_logger=FakeDiagnosticsLogger(),
            memory_log_interval_sec=0.01,
        )
        thread = threading.Thread(target=server.start, daemon=True)

        thread.start()
        assert wait_until(lambda: server.is_running)
        assert wait_until(
            lambda: any(event == "memory_sample" for event, _ in events),
            timeout=3.0,
        )

        server.stop()
        thread.join(timeout=3.0)

        event_names = [event for event, _ in events]
        assert "server_starting" in event_names
        assert "server_started" in event_names
        assert "memory_sample" in event_names
        assert "server_stop_requested" in event_names
        assert "server_stopped" in event_names

    def test_auto_shutdown_stops_server_and_emits_diagnostics(self) -> None:
        events: list[tuple[str, dict[str, object]]] = []

        class FakeDiagnosticsLogger:
            @property
            def path(self) -> str:
                return "/tmp/fake-asee-diagnostics.jsonl"

            def log_event(self, event: str, **fields: object) -> None:
                events.append((event, dict(fields)))

            def open_fault_handler_stream(self) -> None:
                return None

            def close(self) -> None:
                return None

        server = GodModeVideoServer(
            port=18890,
            device_index=None,
            diagnostics_logger=FakeDiagnosticsLogger(),
            memory_log_interval_sec=0.0,
            auto_shutdown_sec=0.2,
        )
        thread = threading.Thread(target=server.start, daemon=True)

        thread.start()
        assert wait_until(
            lambda: any(event == "server_started" for event, _ in events),
            timeout=3.0,
        )
        assert wait_until(
            lambda: any(event == "auto_shutdown_triggered" for event, _ in events),
            timeout=3.0,
        )
        thread.join(timeout=3.0)

        assert server.is_running is False
        event_names = [event for event, _ in events]
        assert "auto_shutdown_armed" in event_names
        assert "auto_shutdown_triggered" in event_names
        assert "server_stopped" in event_names

    def test_server_starts_and_stops(self) -> None:
        server = GodModeVideoServer(port=18871, device_index=None)
        thread = threading.Thread(target=server.start, daemon=True)

        thread.start()
        assert wait_until(lambda: server.is_running)

        server.stop()
        thread.join(timeout=3.0)

        assert server.is_running is False

    def test_http_root_returns_html(self) -> None:
        server = GodModeVideoServer(port=18872, device_index=None)
        thread = threading.Thread(target=server.start, daemon=True)
        thread.start()
        assert wait_until(lambda: server.is_running)

        try:
            with urlopen("http://127.0.0.1:18872/", timeout=3) as response:
                body = response.read().decode("utf-8")
                assert response.status == 200
                assert 'rel="manifest"' in body
                assert "serviceWorker.register" in body
        finally:
            server.stop()
            thread.join(timeout=3.0)

    def test_http_update_endpoint(self) -> None:
        server = GodModeVideoServer(port=18873, device_index=None)
        thread = threading.Thread(target=server.start, daemon=True)
        thread.start()
        assert wait_until(lambda: server.is_running)

        try:
            request = Request(
                "http://127.0.0.1:18873/update",
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
            port=18874,
            device_index=None,
            title="GOD MODE 18873",
        )
        thread = threading.Thread(target=server.start, daemon=True)
        thread.start()
        assert wait_until(lambda: server.is_running)

        try:
            with urlopen(
                "http://127.0.0.1:18874/manifest.webmanifest",
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

    def test_multicamera_update_frame_keeps_primary_as_current_frame(self) -> None:
        server = GodModeVideoServer(
            port=18875,
            device_index=None,
            camera_list=[2, 4],
            allow_live_camera=True,
        )
        secondary = np.zeros((720, 1280, 3), dtype=np.uint8)
        secondary[0, 0, 0] = 50
        primary = np.zeros((720, 1280, 3), dtype=np.uint8)
        primary[0, 0, 0] = 200

        server.update_frame(secondary, camera_id=4)
        assert server.current_frame is None

        server.update_frame(primary, camera_id=2)

        assert server.current_frame is primary

    def test_multicamera_biometric_status_aggregates_across_cameras(self) -> None:
        server = GodModeVideoServer(
            port=18876,
            device_index=None,
            camera_list=[2, 4],
            allow_live_camera=True,
        )

        server._record_owner_presence([FaceBox(x=0, y=0, w=10, h=10, label="OWNER")], camera_id=2)
        server._record_owner_presence(
            [FaceBox(x=20, y=20, w=10, h=10, label="SUBJECT")],
            camera_id=4,
        )
        status = server.get_biometric_status()

        assert status["ownerPresent"] is True
        assert status["ownerCount"] == 1
        assert status["subjectCount"] == 1
        assert status["peopleCount"] == 2

    def test_switch_camera_sets_pending_target_and_event(self) -> None:
        server = GodModeVideoServer(port=18877, device_index=0, allow_live_camera=True)

        server.switch_camera(6)

        assert server._next_device == 6
        assert server._switch_event.is_set() is True

    def test_iter_mjpeg_uses_primary_camera_generator_for_multicamera(self) -> None:
        server = GodModeVideoServer(
            port=18878,
            device_index=None,
            camera_list=[2, 4],
            allow_live_camera=True,
        )

        with (
            patch.object(server, "_generate_mjpeg") as generate_main,
            patch.object(
                server,
                "_generate_mjpeg_device",
                side_effect=["primary-stream", "camera-4-stream"],
            ) as generate_device,
        ):
            primary = server.iter_mjpeg()
            camera_4 = server.iter_mjpeg(4)

        assert primary == "primary-stream"
        assert camera_4 == "camera-4-stream"
        generate_main.assert_not_called()
        assert generate_device.call_args_list == [call(2), call(4)]

    def test_generate_mjpeg_device_uses_face_boxes_for_requested_camera(self) -> None:
        server = GodModeVideoServer(
            port=18879,
            device_index=None,
            camera_list=[2, 4],
            allow_live_camera=True,
        )
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        server.update_frame(frame, camera_id=4)
        face_boxes = [FaceBox(x=30, y=40, w=50, h=60, label="OWNER")]
        server._face_boxes_per_cam[4] = face_boxes

        draw_calls: list[dict[str, object]] = []

        def fake_draw(
            image: np.ndarray,
            *,
            frame_count: int,
            face_boxes: list[FaceBox],
            smooth: bool,
        ) -> np.ndarray:
            draw_calls.append(
                {
                    "frame_count": frame_count,
                    "face_boxes": list(face_boxes),
                    "smooth": smooth,
                }
            )
            server.stop()
            return image

        with patch.object(server.overlay, "draw", side_effect=fake_draw), patch(
            "asee.video_server.encode_frame_to_jpeg",
            return_value=b"jpeg",
        ):
            chunk = next(server._generate_mjpeg_device(4))

        assert chunk.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n\r\njpeg")
        assert draw_calls == [
            {
                "frame_count": 1,
                "face_boxes": face_boxes,
                "smooth": False,
            }
        ]

    def test_http_cameras_endpoint_returns_camera_ids(self) -> None:
        server = GodModeVideoServer(
            port=18880,
            device_index=None,
            camera_list=[2, 4],
            allow_live_camera=True,
        )
        thread = threading.Thread(target=server.start, daemon=True)
        thread.start()
        assert wait_until(lambda: server.is_running)

        try:
            with urlopen("http://127.0.0.1:18880/cameras", timeout=3) as response:
                payload = json.loads(response.read().decode("utf-8"))
                assert response.status == 200
                assert payload == {"cameras": [2, 4]}
        finally:
            server.stop()
            thread.join(timeout=3.0)


class TestOpenCameraResolution:
    def test_decode_fourcc_value_round_trips_mjpg(self) -> None:
        import cv2

        assert decode_fourcc_value(float(cv2.VideoWriter_fourcc(*"MJPG"))) == "MJPG"

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
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 1280.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 720.0,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FOURCC: float(cv2.VideoWriter_fourcc(*"MJPG")),
        }.get(prop, 0.0)

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

"""Unit tests for the extracted GOD MODE video server CLI."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from asee.video_server import (
    CaptureSettings,
    LiveCameraDisabledError,
    build_server_from_args,
    main,
    resolve_camera_args,
    resolve_camera_source_args,
    resolve_capture_settings,
    resolve_default_owner_embedding_path,
    resolve_opencv_threads,
)


def test_resolve_camera_args_returns_none_when_device_is_negative() -> None:
    device_index, camera_list = resolve_camera_args(device=-1, cameras_csv="")

    assert device_index is None
    assert camera_list is None


def test_resolve_camera_args_uses_device_when_camera_csv_is_empty() -> None:
    device_index, camera_list = resolve_camera_args(device=6, cameras_csv="")

    assert device_index == 6
    assert camera_list is None


def test_resolve_camera_args_uses_first_camera_when_csv_is_present() -> None:
    device_index, camera_list = resolve_camera_args(device=6, cameras_csv="2,4,6")

    assert device_index == 2
    assert camera_list == [2, 4, 6]


def test_resolve_camera_source_args_parses_mixed_usb_and_rtsp_sources() -> None:
    device_index, camera_list, camera_source_map = resolve_camera_source_args(
        camera_sources_csv=(
            "0@0,"
            "2@2,"
            "4@rtsp://atomcam-hoge.local:8554/video0_unicast,"
            "6@rtsp://atomcam-fuga.local:8554/video0_unicast"
        )
    )

    assert device_index == 0
    assert camera_list == [0, 2, 4, 6]
    assert camera_source_map == {
        0: 0,
        2: 2,
        4: "rtsp://atomcam-hoge.local:8554/video0_unicast",
        6: "rtsp://atomcam-fuga.local:8554/video0_unicast",
    }


def test_resolve_capture_settings_defaults_multicamera_to_720p_30fps() -> None:
    settings = resolve_capture_settings(camera_ids=[0, 2, 4, 6])

    assert settings == CaptureSettings(
        width=1280,
        height=720,
        fps=30.0,
        fourcc="MJPG",
    )


def test_resolve_capture_settings_supports_720p_profile_for_multicamera() -> None:
    settings = resolve_capture_settings(camera_ids=[0, 2, 4, 6], capture_profile="720p")

    assert settings == CaptureSettings(
        width=1280,
        height=720,
        fps=30.0,
        fourcc="MJPG",
    )


def test_resolve_capture_settings_respects_explicit_overrides() -> None:
    settings = resolve_capture_settings(
        camera_ids=[0, 2],
        capture_profile="720p",
        width=800,
        height=600,
        fps=12.5,
        fourcc="YUYV",
    )

    assert settings == CaptureSettings(
        width=800,
        height=600,
        fps=12.5,
        fourcc="YUYV",
    )


def test_resolve_opencv_threads_uses_safe_multicamera_default() -> None:
    assert resolve_opencv_threads(camera_ids=[0, 2, 4, 6]) == 1


def test_resolve_opencv_threads_respects_explicit_override() -> None:
    assert resolve_opencv_threads(camera_ids=[0, 2], opencv_threads=3) == 3


def test_build_server_from_args_disables_empty_capture_dirs() -> None:
    args = SimpleNamespace(
        port=8765,
        device=0,
        cameras="",
        camera_sources="",
        cam_interval=60,
        title="GOD MODE",
        face_capture_dir="",
        face_capture_min_interval=1.5,
        subject_capture_dir="",
        full_frame_capture_dir="",
        full_frame_morning_interval_sec=300.0,
        full_frame_day_interval_sec=900.0,
        full_frame_evening_interval_sec=600.0,
        full_frame_overnight_interval_sec=0.0,
        allow_live_camera=True,
        diagnostic_log_path="/tmp/asee-test.jsonl",
        memory_log_interval_sec=12.5,
        auto_shutdown_sec=90.0,
        capture_profile="auto",
        motion_sensor_name="リビングルームの人感センサー",
        meter_name="リビング温湿度計",
        room_context_ttl_sec=5.0,
        width=None,
        height=None,
        fps=None,
        fourcc=None,
        opencv_threads=None,
        disable_face_detect=True,
        detection_backend="opencv",
        recognition_backend="facenet-pytorch",
        transport="webrtc",
    )

    with (
        patch("asee.video_server.GodModeVideoServer") as server_class,
        patch("asee.video_server.JsonlDiagnosticsLogger") as logger_class,
        patch(
            "asee.video_server.SwitchBotRoomContextProvider",
            return_value="room-context-provider",
        ),
    ):
        build_server_from_args(args)

    server_class.assert_called_once_with(
        port=8765,
        device_index=0,
        camera_list=None,
        camera_source_map=None,
        cam_interval=60,
        title="GOD MODE",
        face_capture_dir=None,
        face_capture_min_interval=1.5,
        subject_capture_dir=None,
        full_frame_capture_dir=None,
        full_frame_morning_interval_sec=300.0,
        full_frame_day_interval_sec=900.0,
        full_frame_evening_interval_sec=600.0,
        full_frame_overnight_interval_sec=0.0,
        allow_live_camera=True,
        diagnostics_logger=logger_class.return_value,
        memory_log_interval_sec=12.5,
        auto_shutdown_sec=90.0,
        capture_profile="auto",
        room_context_provider="room-context-provider",
        width=None,
        height=None,
        fps=None,
        fourcc=None,
        opencv_threads=None,
        enable_face_detection=False,
        detection_backend="opencv",
        insightface_det_size=320,
        recognition_backend="facenet-pytorch",
        owner_embedding_path=resolve_default_owner_embedding_path("facenet-pytorch"),
        transport="webrtc",
    )


def test_build_server_from_args_defaults_to_persistent_diagnostics_log_path() -> None:
    args = SimpleNamespace(
        port=8765,
        device=-1,
        cameras="",
        camera_sources="",
        cam_interval=60,
        title="GOD MODE",
        face_capture_dir="",
        face_capture_min_interval=1.5,
        subject_capture_dir="",
        full_frame_capture_dir="",
        full_frame_morning_interval_sec=300.0,
        full_frame_day_interval_sec=900.0,
        full_frame_evening_interval_sec=600.0,
        full_frame_overnight_interval_sec=0.0,
        allow_live_camera=False,
        diagnostic_log_path=None,
        memory_log_interval_sec=30.0,
        auto_shutdown_sec=0.0,
        capture_profile="auto",
        motion_sensor_name="リビングルームの人感センサー",
        meter_name="リビング温湿度計",
        room_context_ttl_sec=5.0,
        width=None,
        height=None,
        fps=None,
        fourcc=None,
        opencv_threads=None,
        disable_face_detect=False,
        detection_backend="opencv",
        recognition_backend="facenet-pytorch",
        transport="webrtc",
    )

    with (
        patch("asee.video_server.GodModeVideoServer") as server_class,
        patch("asee.video_server.JsonlDiagnosticsLogger") as logger_class,
        patch(
            "asee.video_server.SwitchBotRoomContextProvider",
            return_value="room-context-provider",
        ),
        patch(
            "asee.video_server.build_default_diagnostics_log_path",
            return_value=Path("/tmp/default-asee.jsonl"),
        ) as build_path,
    ):
        build_server_from_args(args)

    build_path.assert_called_once_with()
    logger_class.assert_called_once_with(Path("/tmp/default-asee.jsonl"))
    server_class.assert_called_once()


def test_build_server_from_args_rejects_live_camera_without_explicit_opt_in() -> None:
    args = SimpleNamespace(
        port=8765,
        device=0,
        cameras="",
        camera_sources="",
        cam_interval=60,
        title="GOD MODE",
        face_capture_dir="",
        face_capture_min_interval=1.5,
        subject_capture_dir="",
        full_frame_capture_dir="",
        full_frame_morning_interval_sec=300.0,
        full_frame_day_interval_sec=900.0,
        full_frame_evening_interval_sec=600.0,
        full_frame_overnight_interval_sec=0.0,
        allow_live_camera=False,
        diagnostic_log_path="/tmp/asee-test.jsonl",
        memory_log_interval_sec=12.5,
        auto_shutdown_sec=0.0,
        capture_profile="auto",
        motion_sensor_name="リビングルームの人感センサー",
        meter_name="リビング温湿度計",
        room_context_ttl_sec=5.0,
        width=None,
        height=None,
        fps=None,
        fourcc=None,
        opencv_threads=None,
        disable_face_detect=False,
        detection_backend="opencv",
        recognition_backend="facenet-pytorch",
        transport="webrtc",
    )

    with pytest.raises(LiveCameraDisabledError, match="allow-live-camera"):
        build_server_from_args(args)


def test_build_server_from_args_prefers_explicit_camera_sources() -> None:
    args = SimpleNamespace(
        port=8765,
        device=0,
        cameras="0,2,4,6",
        camera_sources=(
            "0@0,"
            "2@2,"
            "4@rtsp://atomcam-hoge.local:8554/video0_unicast,"
            "6@rtsp://atomcam-fuga.local:8554/video0_unicast"
        ),
        cam_interval=60,
        title="GOD MODE",
        face_capture_dir="",
        face_capture_min_interval=1.0,
        subject_capture_dir="",
        full_frame_capture_dir="",
        full_frame_morning_interval_sec=300.0,
        full_frame_day_interval_sec=900.0,
        full_frame_evening_interval_sec=600.0,
        full_frame_overnight_interval_sec=0.0,
        allow_live_camera=True,
        diagnostic_log_path="/tmp/asee-test.jsonl",
        memory_log_interval_sec=30.0,
        auto_shutdown_sec=0.0,
        capture_profile="auto",
        motion_sensor_name="",
        meter_name="",
        room_context_ttl_sec=5.0,
        width=None,
        height=None,
        fps=None,
        fourcc=None,
        opencv_threads=None,
        disable_face_detect=False,
        detection_backend="insightface",
        insightface_det_size=320,
        recognition_backend="facenet-pytorch",
        transport="webrtc",
    )

    with (
        patch("asee.video_server.GodModeVideoServer") as server_class,
        patch("asee.video_server.JsonlDiagnosticsLogger"),
        patch("asee.video_server.SwitchBotRoomContextProvider"),
    ):
        build_server_from_args(args)

    call_kwargs = server_class.call_args.kwargs
    assert call_kwargs["device_index"] == 0
    assert call_kwargs["camera_list"] == [0, 2, 4, 6]
    assert call_kwargs["camera_source_map"] == {
        0: 0,
        2: 2,
        4: "rtsp://atomcam-hoge.local:8554/video0_unicast",
        6: "rtsp://atomcam-fuga.local:8554/video0_unicast",
    }


def test_build_arg_parser_accepts_detection_backend_onnxruntime() -> None:
    """--detection-backend onnxruntime must be accepted by the arg parser."""
    from asee.video_server import build_arg_parser

    parser = build_arg_parser()
    args = parser.parse_args(["--detection-backend", "onnxruntime"])
    assert args.detection_backend == "onnxruntime"


def test_build_arg_parser_accepts_detection_backend_insightface() -> None:
    from asee.video_server import build_arg_parser

    parser = build_arg_parser()
    args = parser.parse_args(["--detection-backend", "insightface"])
    assert args.detection_backend == "insightface"


def test_build_arg_parser_accepts_camera_sources() -> None:
    from asee.video_server import build_arg_parser

    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--camera-sources",
            "0@0,2@2,4@rtsp://atomcam-hoge.local:8554/video0_unicast",
        ]
    )
    assert args.camera_sources == "0@0,2@2,4@rtsp://atomcam-hoge.local:8554/video0_unicast"


def test_build_arg_parser_accepts_recognition_backend_opencv_sface() -> None:
    from asee.video_server import build_arg_parser

    parser = build_arg_parser()
    args = parser.parse_args(["--recognition-backend", "opencv-sface"])
    assert args.recognition_backend == "opencv-sface"


def test_build_arg_parser_accepts_full_frame_capture_dir() -> None:
    from asee.video_server import build_arg_parser

    parser = build_arg_parser()
    args = parser.parse_args(["--full-frame-capture-dir", "/tmp/webcams"])
    assert args.full_frame_capture_dir == "/tmp/webcams"


def test_build_arg_parser_full_frame_sampling_defaults_are_conservative() -> None:
    from asee.video_server import build_arg_parser

    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.full_frame_morning_interval_sec == 300.0
    assert args.full_frame_day_interval_sec == 900.0
    assert args.full_frame_evening_interval_sec == 600.0
    assert args.full_frame_overnight_interval_sec == 0.0


def test_build_arg_parser_detection_backend_default_is_insightface() -> None:
    """--detection-backend must default to 'insightface'."""
    from asee.video_server import build_arg_parser

    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.detection_backend == "insightface"


def test_build_arg_parser_insightface_det_size_default_is_320() -> None:
    from asee.video_server import build_arg_parser

    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.insightface_det_size == 320


def test_build_arg_parser_recognition_backend_default_is_facenet_pytorch() -> None:
    from asee.video_server import build_arg_parser

    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.recognition_backend == "facenet-pytorch"


def test_resolve_default_owner_embedding_path_for_facenet() -> None:
    path = resolve_default_owner_embedding_path("facenet-pytorch")

    assert path.name == "owner_embedding_facenet_pytorch.npy"


def test_resolve_default_owner_embedding_path_for_opencv_sface() -> None:
    path = resolve_default_owner_embedding_path("opencv-sface")

    assert path.name == "owner_embedding_opencv_sface.npy"


def test_build_arg_parser_transport_default_is_webrtc() -> None:
    from asee.video_server import build_arg_parser

    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.transport == "webrtc"


def test_build_server_from_args_passes_detection_backend_to_server() -> None:
    """build_server_from_args must forward detection_backend to GodModeVideoServer."""
    args = SimpleNamespace(
        port=8765,
        device=-1,
        cameras="",
        cam_interval=60,
        title="GOD MODE",
        face_capture_dir="",
        face_capture_min_interval=1.0,
        subject_capture_dir="",
        full_frame_capture_dir="/tmp/webcams",
        full_frame_morning_interval_sec=120.0,
        full_frame_day_interval_sec=600.0,
        full_frame_evening_interval_sec=480.0,
        full_frame_overnight_interval_sec=0.0,
        allow_live_camera=False,
        diagnostic_log_path="/tmp/asee-test.jsonl",
        memory_log_interval_sec=30.0,
        auto_shutdown_sec=0.0,
        capture_profile="auto",
        motion_sensor_name="リビングルームの人感センサー",
        meter_name="リビング温湿度計",
        room_context_ttl_sec=5.0,
        width=None,
        height=None,
        fps=None,
        fourcc=None,
        opencv_threads=None,
        disable_face_detect=False,
        detection_backend="onnxruntime",
        insightface_det_size=320,
        recognition_backend="opencv-sface",
        transport="webrtc",
    )

    with (
        patch("asee.video_server.GodModeVideoServer") as server_class,
        patch("asee.video_server.JsonlDiagnosticsLogger"),
        patch(
            "asee.video_server.SwitchBotRoomContextProvider",
            return_value="room-context-provider",
        ),
    ):
        build_server_from_args(args)

    call_kwargs = server_class.call_args.kwargs
    assert call_kwargs.get("detection_backend") == "onnxruntime"
    assert call_kwargs.get("insightface_det_size") == 320
    assert call_kwargs.get("recognition_backend") == "opencv-sface"
    assert call_kwargs.get("owner_embedding_path").name == "owner_embedding_opencv_sface.npy"
    assert call_kwargs.get("transport") == "webrtc"
    assert call_kwargs.get("full_frame_capture_dir") == "/tmp/webcams"
    assert call_kwargs.get("full_frame_morning_interval_sec") == 120.0
    assert call_kwargs.get("full_frame_day_interval_sec") == 600.0
    assert call_kwargs.get("full_frame_evening_interval_sec") == 480.0
    assert call_kwargs.get("full_frame_overnight_interval_sec") == 0.0


def test_build_server_from_args_defaults_facenet_owner_embedding_path() -> None:
    args = SimpleNamespace(
        port=8765,
        device=-1,
        cameras="",
        cam_interval=60,
        title="GOD MODE",
        face_capture_dir="",
        face_capture_min_interval=1.0,
        subject_capture_dir="",
        full_frame_capture_dir="",
        full_frame_morning_interval_sec=300.0,
        full_frame_day_interval_sec=900.0,
        full_frame_evening_interval_sec=600.0,
        full_frame_overnight_interval_sec=0.0,
        allow_live_camera=False,
        diagnostic_log_path="/tmp/asee-test.jsonl",
        memory_log_interval_sec=30.0,
        auto_shutdown_sec=0.0,
        capture_profile="auto",
        motion_sensor_name="リビングルームの人感センサー",
        meter_name="リビング温湿度計",
        room_context_ttl_sec=5.0,
        width=None,
        height=None,
        fps=None,
        fourcc=None,
        opencv_threads=None,
        disable_face_detect=False,
        detection_backend="insightface",
        insightface_det_size=320,
        recognition_backend="facenet-pytorch",
        transport="webrtc",
    )

    with (
        patch("asee.video_server.GodModeVideoServer") as server_class,
        patch("asee.video_server.JsonlDiagnosticsLogger"),
        patch(
            "asee.video_server.SwitchBotRoomContextProvider",
            return_value="room-context-provider",
        ),
    ):
        build_server_from_args(args)

    call_kwargs = server_class.call_args.kwargs
    assert call_kwargs.get("detection_backend") == "insightface"
    assert call_kwargs.get("insightface_det_size") == 320
    assert call_kwargs.get("recognition_backend") == "facenet-pytorch"
    assert call_kwargs.get("owner_embedding_path").name == "owner_embedding_facenet_pytorch.npy"


def test_build_server_from_args_disables_room_context_when_sensor_names_are_empty() -> None:
    args = SimpleNamespace(
        port=8765,
        device=-1,
        cameras="",
        cam_interval=60,
        title="GOD MODE",
        face_capture_dir="",
        face_capture_min_interval=1.0,
        subject_capture_dir="",
        full_frame_capture_dir="",
        full_frame_morning_interval_sec=300.0,
        full_frame_day_interval_sec=900.0,
        full_frame_evening_interval_sec=600.0,
        full_frame_overnight_interval_sec=0.0,
        allow_live_camera=False,
        diagnostic_log_path="/tmp/asee-test.jsonl",
        memory_log_interval_sec=30.0,
        auto_shutdown_sec=0.0,
        capture_profile="auto",
        motion_sensor_name="",
        meter_name="",
        room_context_ttl_sec=5.0,
        width=None,
        height=None,
        fps=None,
        fourcc=None,
        opencv_threads=None,
        disable_face_detect=False,
        detection_backend="onnxruntime",
        recognition_backend="facenet-pytorch",
        transport="webrtc",
    )

    with (
        patch("asee.video_server.GodModeVideoServer") as server_class,
        patch("asee.video_server.JsonlDiagnosticsLogger"),
        patch("asee.video_server.SwitchBotRoomContextProvider") as provider_class,
    ):
        build_server_from_args(args)

    provider_class.assert_not_called()
    assert server_class.call_args.kwargs["room_context_provider"] is None


def test_main_builds_server_and_starts_it() -> None:
    started = []

    class FakeServer:
        def start(self) -> None:
            started.append("started")

    with patch("asee.video_server.build_server_from_args", return_value=FakeServer()) as builder:
        exit_code = main(
            [
                "--port",
                "9000",
                "--device",
                "4",
            ]
        )

    assert exit_code == 0
    builder.assert_called_once()
    assert started == ["started"]

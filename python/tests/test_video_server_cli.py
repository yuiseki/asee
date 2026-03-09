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
    resolve_capture_settings,
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


def test_resolve_capture_settings_uses_safe_multicamera_defaults() -> None:
    settings = resolve_capture_settings(camera_ids=[0, 2, 4, 6])

    assert settings == CaptureSettings(
        width=640,
        height=360,
        fps=10.0,
        fourcc="MJPG",
    )


def test_resolve_capture_settings_supports_720p_profile_for_multicamera() -> None:
    settings = resolve_capture_settings(camera_ids=[0, 2, 4, 6], capture_profile="720p")

    assert settings == CaptureSettings(
        width=1280,
        height=720,
        fps=10.0,
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
        cam_interval=60,
        title="GOD MODE",
        face_capture_dir="",
        face_capture_min_interval=1.5,
        subject_capture_dir="",
        allow_live_camera=True,
        diagnostic_log_path="/tmp/asee-test.jsonl",
        memory_log_interval_sec=12.5,
        auto_shutdown_sec=90.0,
        capture_profile="auto",
        width=None,
        height=None,
        fps=None,
        fourcc=None,
        opencv_threads=None,
        disable_face_detect=True,
        detection_backend="opencv",
    )

    with (
        patch("asee.video_server.GodModeVideoServer") as server_class,
        patch("asee.video_server.JsonlDiagnosticsLogger") as logger_class,
    ):
        build_server_from_args(args)

    server_class.assert_called_once_with(
        port=8765,
        device_index=0,
        camera_list=None,
        cam_interval=60,
        title="GOD MODE",
        face_capture_dir=None,
        face_capture_min_interval=1.5,
        subject_capture_dir=None,
        allow_live_camera=True,
        diagnostics_logger=logger_class.return_value,
        memory_log_interval_sec=12.5,
        auto_shutdown_sec=90.0,
        capture_profile="auto",
        width=None,
        height=None,
        fps=None,
        fourcc=None,
        opencv_threads=None,
        enable_face_detection=False,
        detection_backend="opencv",
    )


def test_build_server_from_args_defaults_to_persistent_diagnostics_log_path() -> None:
    args = SimpleNamespace(
        port=8765,
        device=-1,
        cameras="",
        cam_interval=60,
        title="GOD MODE",
        face_capture_dir="",
        face_capture_min_interval=1.5,
        subject_capture_dir="",
        allow_live_camera=False,
        diagnostic_log_path=None,
        memory_log_interval_sec=30.0,
        auto_shutdown_sec=0.0,
        capture_profile="auto",
        width=None,
        height=None,
        fps=None,
        fourcc=None,
        opencv_threads=None,
        disable_face_detect=False,
        detection_backend="opencv",
    )

    with (
        patch("asee.video_server.GodModeVideoServer") as server_class,
        patch("asee.video_server.JsonlDiagnosticsLogger") as logger_class,
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
        cam_interval=60,
        title="GOD MODE",
        face_capture_dir="",
        face_capture_min_interval=1.5,
        subject_capture_dir="",
        allow_live_camera=False,
        diagnostic_log_path="/tmp/asee-test.jsonl",
        memory_log_interval_sec=12.5,
        auto_shutdown_sec=0.0,
        capture_profile="auto",
        width=None,
        height=None,
        fps=None,
        fourcc=None,
        opencv_threads=None,
        disable_face_detect=False,
        detection_backend="opencv",
    )

    with pytest.raises(LiveCameraDisabledError, match="allow-live-camera"):
        build_server_from_args(args)


def test_build_arg_parser_accepts_detection_backend_onnxruntime() -> None:
    """--detection-backend onnxruntime must be accepted by the arg parser."""
    from asee.video_server import build_arg_parser

    parser = build_arg_parser()
    args = parser.parse_args(["--detection-backend", "onnxruntime"])
    assert args.detection_backend == "onnxruntime"


def test_build_arg_parser_detection_backend_default_is_opencv() -> None:
    """--detection-backend must default to 'opencv'."""
    from asee.video_server import build_arg_parser

    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.detection_backend == "opencv"


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
        allow_live_camera=False,
        diagnostic_log_path="/tmp/asee-test.jsonl",
        memory_log_interval_sec=30.0,
        auto_shutdown_sec=0.0,
        capture_profile="auto",
        width=None,
        height=None,
        fps=None,
        fourcc=None,
        opencv_threads=None,
        disable_face_detect=False,
        detection_backend="onnxruntime",
    )

    with (
        patch("asee.video_server.GodModeVideoServer") as server_class,
        patch("asee.video_server.JsonlDiagnosticsLogger"),
    ):
        build_server_from_args(args)

    call_kwargs = server_class.call_args.kwargs
    assert call_kwargs.get("detection_backend") == "onnxruntime"


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

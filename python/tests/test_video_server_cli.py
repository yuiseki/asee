"""Unit tests for the extracted GOD MODE video server CLI."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from asee.video_server import build_server_from_args, main, resolve_camera_args


def test_resolve_camera_args_uses_device_when_camera_csv_is_empty() -> None:
    device_index, camera_list = resolve_camera_args(device=6, cameras_csv="")

    assert device_index == 6
    assert camera_list is None


def test_resolve_camera_args_uses_first_camera_when_csv_is_present() -> None:
    device_index, camera_list = resolve_camera_args(device=6, cameras_csv="2,4,6")

    assert device_index == 2
    assert camera_list == [2, 4, 6]


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
    )

    with patch("asee.video_server.GodModeVideoServer") as server_class:
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
    )


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

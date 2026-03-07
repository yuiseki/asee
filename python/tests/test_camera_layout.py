from unittest.mock import patch

from asee.camera_layout import (
    build_camera_csv,
    extend_with_optional_camera,
    main,
    parse_camera_csv,
    parse_v4l2_devices,
)

SAMPLE_V4L2 = """
HD Pro Webcam C920 (usb-0000:06:00.1-5):
\t/dev/video0
\t/dev/video1
\t/dev/media0

HD webcam-CMS-V43BK: HD webcam- (usb-0000:06:00.3-4):
\t/dev/video2
\t/dev/video3
\t/dev/media1

HD Pro Webcam C920 (usb-0000:06:00.3-5.3):
\t/dev/video4
\t/dev/video5
\t/dev/media2

Anker PowerConf C200 (usb-0000:06:00.3-6):
\t/dev/video6
\t/dev/video7
\t/dev/media3
""".strip()


def test_parse_v4l2_devices_uses_first_video_node_per_device() -> None:
    devices = parse_v4l2_devices(SAMPLE_V4L2)
    assert devices == [
        ("HD Pro Webcam C920 (usb-0000:06:00.1-5)", 0),
        ("HD webcam-CMS-V43BK: HD webcam- (usb-0000:06:00.3-4)", 2),
        ("HD Pro Webcam C920 (usb-0000:06:00.3-5.3)", 4),
        ("Anker PowerConf C200 (usb-0000:06:00.3-6)", 6),
    ]


def test_extend_with_optional_camera_keeps_base_when_no_extra_exists() -> None:
    devices = parse_v4l2_devices(SAMPLE_V4L2)
    assert extend_with_optional_camera([0, 2, 4, 6], devices) == [0, 2, 4, 6]


def test_extend_with_optional_camera_appends_first_extra_device() -> None:
    devices = parse_v4l2_devices(SAMPLE_V4L2)
    assert extend_with_optional_camera([0, 2, 4], devices, preferred_tokens=()) == [0, 2, 4, 6]


def test_extend_with_optional_camera_prefers_anker_named_device() -> None:
    devices = [
        ("Some Generic Cam", 6),
        ("Anker PowerConf C200", 8),
    ]
    assert extend_with_optional_camera(
        [0, 2, 4],
        devices,
        preferred_tokens=("anker",),
    ) == [0, 2, 4, 8]


def test_parse_camera_csv_ignores_empty_parts() -> None:
    assert parse_camera_csv("0, 2, , 4") == [0, 2, 4]


def test_build_camera_csv_appends_detected_optional_camera() -> None:
    with patch(
        "asee.camera_layout.detect_v4l2_devices",
        return_value=parse_v4l2_devices(SAMPLE_V4L2),
    ):
        result = build_camera_csv("0,2,4", ("anker",))

    assert result == "0,2,4,6"


def test_main_prints_computed_camera_csv(capsys) -> None:
    with (
        patch(
            "asee.camera_layout.detect_v4l2_devices",
            return_value=parse_v4l2_devices(SAMPLE_V4L2),
        ),
        patch("sys.argv", ["asee.camera_layout", "--base", "0,2,4", "--prefer", "anker"]),
    ):
        exit_code = main()

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == "0,2,4,6"

"""Unit tests for OWNER enrollment helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from asee.enroll_owner import (
    EnrollmentError,
    fetch_frame_from_server,
    main,
    run_enrollment,
)


def test_fetch_frame_from_server_decodes_snapshot() -> None:
    response = SimpleNamespace(status=200, read=lambda: b"jpeg-bytes")
    with (
        patch("asee.enroll_owner.urlopen") as urlopen_mock,
        patch("asee.enroll_owner.cv2.imdecode", return_value="decoded-frame") as imdecode_mock,
    ):
        urlopen_mock.return_value.__enter__.return_value = response

        result = fetch_frame_from_server("http://localhost:8765")

    assert result == "decoded-frame"
    imdecode_mock.assert_called_once()


def test_fetch_frame_from_server_returns_none_on_error() -> None:
    with patch("asee.enroll_owner.urlopen", side_effect=OSError("down")):
        result = fetch_frame_from_server("http://localhost:8765")

    assert result is None


class FakeOverlay:
    def __init__(self) -> None:
        self._detector = object()
        self._recognizer = object()
        self._embeddings = [
            np.ones((1, 128), dtype=np.float32),
            np.ones((1, 128), dtype=np.float32) * 2,
        ]

    def detect_faces(self, frame: np.ndarray) -> list[SimpleNamespace]:
        del frame
        return [SimpleNamespace(w=100, h=100, confidence=0.9)]

    def extract_embedding(
        self,
        frame: np.ndarray,
        face_box: SimpleNamespace,
    ) -> np.ndarray | None:
        del frame, face_box
        if not self._embeddings:
            return None
        return self._embeddings.pop(0)


def test_run_enrollment_uses_server_snapshot_without_opening_camera(tmp_path: Path) -> None:
    overlay = FakeOverlay()
    frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(3)]
    saved = []

    def fake_fetch_frame(server_url: str) -> np.ndarray | None:
        del server_url
        if frames:
            return frames.pop(0)
        return None

    def fake_save(path: Path, data: np.ndarray) -> None:
        saved.append((path, data.copy()))

    run_enrollment(
        device_index=0,
        n_samples=2,
        server_url="http://localhost:8765",
        save_path=tmp_path / "owner.npy",
        overlay=overlay,
        fetch_frame=fake_fetch_frame,
        save_embeddings=fake_save,
        sleep=lambda _seconds: None,
        video_capture_factory=Mock(side_effect=AssertionError("camera should not open")),
    )

    assert len(saved) == 1
    save_path, stacked = saved[0]
    assert save_path == tmp_path / "owner.npy"
    assert stacked.shape == (2, 1, 128)


def test_run_enrollment_falls_back_to_direct_camera(tmp_path: Path) -> None:
    overlay = FakeOverlay()
    read_frames = [
        (True, np.zeros((720, 1280, 3), dtype=np.uint8)),
        (True, np.zeros((720, 1280, 3), dtype=np.uint8)),
    ]
    camera = Mock()
    camera.isOpened.return_value = True
    camera.read.side_effect = read_frames
    saved = []

    run_enrollment(
        device_index=4,
        n_samples=2,
        server_url="http://localhost:8765",
        save_path=tmp_path / "owner.npy",
        overlay=overlay,
        fetch_frame=lambda _url: None,
        save_embeddings=lambda path, data: saved.append((path, data.copy())),
        sleep=lambda _seconds: None,
        video_capture_factory=Mock(return_value=camera),
    )

    assert camera.read.call_count >= 2
    camera.release.assert_called_once()
    assert saved[0][1].shape == (2, 1, 128)


def test_run_enrollment_raises_when_not_enough_samples(tmp_path: Path) -> None:
    overlay = FakeOverlay()
    overlay.detect_faces = lambda _frame: []

    with pytest.raises(EnrollmentError, match="Not enough samples"):
        run_enrollment(
            device_index=0,
            n_samples=4,
            server_url="",
            save_path=tmp_path / "owner.npy",
            overlay=overlay,
            fetch_frame=lambda _url: None,
            save_embeddings=lambda _path, _data: None,
            sleep=lambda _seconds: None,
            video_capture_factory=Mock(
                return_value=Mock(
                    isOpened=Mock(return_value=True),
                    read=Mock(return_value=(False, None)),
                    release=Mock(),
                )
            ),
        )


def test_main_returns_zero_on_success(tmp_path: Path) -> None:
    with patch(
        "asee.enroll_owner.run_enrollment",
        return_value=np.zeros((2, 1, 128), dtype=np.float32),
    ) as run_mock:
        exit_code = main(
            [
                "--device",
                "4",
                "--samples",
                "2",
                "--server",
                "http://localhost:8765",
                "--save-path",
                str(tmp_path / "owner.npy"),
            ]
        )

    assert exit_code == 0
    run_mock.assert_called_once()

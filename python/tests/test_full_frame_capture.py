"""Unit tests for periodically sampled full-frame capture persistence."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from asee.full_frame_capture import FullFrameCaptureWriter


def write_dummy_image(path: Path, image: object) -> bool:
    del image
    path.write_bytes(b"jpg")
    return True


class Clock:
    def __init__(self, start: float) -> None:
        self.current = start

    def __call__(self) -> float:
        return self.current


class Dates:
    def __init__(self, *values: datetime) -> None:
        self._values = list(values)

    def __call__(self) -> datetime:
        if len(self._values) == 1:
            return self._values[0]
        return self._values.pop(0)


def test_save_creates_video_camera_partition(tmp_path: Path) -> None:
    writer = FullFrameCaptureWriter(
        tmp_path,
        write_image=write_dummy_image,
        now_provider=Dates(datetime(2026, 3, 29, 8, 12, 34)),
        time_provider=Clock(1000.0),
    )

    saved = writer.save(object(), camera_id=7)

    assert saved is True
    jpgs = list(tmp_path.rglob("*.jpg"))
    assert len(jpgs) == 1
    rel = jpgs[0].relative_to(tmp_path)
    assert rel.parts[:6] == ("video7", "2026", "03", "29", "08", "12")
    assert rel.name == "frame-20260329-081234-0001.jpg"


def test_save_writes_presence_sidecar_metadata(tmp_path: Path) -> None:
    writer = FullFrameCaptureWriter(
        tmp_path,
        write_image=write_dummy_image,
        now_provider=Dates(datetime(2026, 3, 29, 16, 5, 0)),
        time_provider=Clock(2000.0),
    )

    saved = writer.save(
        object(),
        camera_id=2,
        metadata={
            "presence": {
                "cameraPeopleCount": 1,
                "cameraOwnerCount": 1,
                "cameraSubjectCount": 0,
                "globalOwnerPresent": True,
            },
            "width": 1280,
            "height": 720,
        },
    )

    assert saved is True
    sidecar = list(tmp_path.rglob("*.json"))[0]
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["cameraId"] == 2
    assert payload["schedulePeriod"] == "day"
    assert payload["presence"] == {
        "cameraPeopleCount": 1,
        "cameraOwnerCount": 1,
        "cameraSubjectCount": 0,
        "globalOwnerPresent": True,
    }
    assert payload["width"] == 1280
    assert payload["height"] == 720
    assert payload["capturedAt"] == "2026-03-29T16:05:00"


def test_save_uses_per_period_intervals_per_camera(tmp_path: Path) -> None:
    clock = Clock(1000.0)
    writer = FullFrameCaptureWriter(
        tmp_path,
        morning_interval_sec=300.0,
        write_image=write_dummy_image,
        now_provider=Dates(
            datetime(2026, 3, 29, 8, 0, 0),
            datetime(2026, 3, 29, 8, 1, 0),
            datetime(2026, 3, 29, 8, 6, 0),
        ),
        time_provider=clock,
    )

    assert writer.save(object(), camera_id=0) is True
    clock.current += 60.0
    assert writer.save(object(), camera_id=0) is False
    clock.current += 300.0
    assert writer.save(object(), camera_id=0) is True
    assert len(list(tmp_path.rglob("*.jpg"))) == 2


def test_save_is_disabled_overnight_by_default(tmp_path: Path) -> None:
    writer = FullFrameCaptureWriter(
        tmp_path,
        write_image=write_dummy_image,
        now_provider=Dates(datetime(2026, 3, 29, 2, 30, 0)),
        time_provider=Clock(100.0),
    )

    assert writer.save(object(), camera_id=4) is False
    assert list(tmp_path.rglob("*.jpg")) == []

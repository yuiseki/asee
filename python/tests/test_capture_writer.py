"""Unit tests for extracted face capture persistence."""

from __future__ import annotations

import json
from pathlib import Path

from asee.capture_writer import FaceCaptureWriter


def write_dummy_image(path: Path, image: object) -> bool:
    del image
    path.write_bytes(b'jpg')
    return True


def test_save_creates_file_in_yyyy_mm_dd_hh_mm_subdir(tmp_path):
    writer = FaceCaptureWriter(
        tmp_path,
        min_interval_sec=0.0,
        write_image=write_dummy_image,
    )

    saved = writer.save(object(), score=0.75)

    assert saved is True
    jpgs = list(tmp_path.rglob('*.jpg'))
    assert len(jpgs) == 1
    rel = jpgs[0].relative_to(tmp_path)
    parts = rel.parts
    assert len(parts) == 6
    assert len(parts[0]) == 4 and parts[0].isdigit()
    assert len(parts[1]) == 2 and parts[1].isdigit()
    assert len(parts[2]) == 2 and parts[2].isdigit()
    assert len(parts[3]) == 2 and parts[3].isdigit()
    assert len(parts[4]) == 2 and parts[4].isdigit()


def test_min_interval_prevents_rapid_saves(tmp_path):
    writer = FaceCaptureWriter(
        tmp_path,
        min_interval_sec=10.0,
        write_image=write_dummy_image,
    )

    first = writer.save(object(), score=0.80)
    second = writer.save(object(), score=0.80)

    assert first is True
    assert second is False
    assert len(list(tmp_path.rglob('*.jpg'))) == 1


def test_score_is_in_filename(tmp_path):
    writer = FaceCaptureWriter(
        tmp_path,
        min_interval_sec=0.0,
        write_image=write_dummy_image,
    )

    writer.save(object(), score=0.63)

    jpg = list(tmp_path.rglob('*.jpg'))[0]
    assert 'score0.63' in jpg.name


def test_save_writes_sidecar_metadata_json(tmp_path):
    writer = FaceCaptureWriter(
        tmp_path,
        min_interval_sec=0.0,
        write_image=write_dummy_image,
    )

    saved = writer.save(
        object(),
        score=0.63,
        metadata={
            "label": "SUBJECT",
            "cameraId": 2,
            "frameCounts": {"ownerCount": 0, "subjectCount": 1, "peopleCount": 1},
        },
    )

    assert saved is True
    jpg = list(tmp_path.rglob('*.jpg'))[0]
    sidecar = jpg.with_suffix(".json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["score"] == 0.63
    assert payload["label"] == "SUBJECT"
    assert payload["cameraId"] == 2
    assert payload["frameCounts"] == {"ownerCount": 0, "subjectCount": 1, "peopleCount": 1}
    assert "capturedAt" in payload


def test_max_files_per_day_limit(tmp_path):
    writer = FaceCaptureWriter(
        tmp_path,
        min_interval_sec=0.0,
        max_files_per_day=3,
        write_image=write_dummy_image,
    )

    results = [writer.save(object(), score=0.7) for _ in range(5)]

    assert results[:3] == [True, True, True]
    assert results[3] is False
    assert results[4] is False
    assert len(list(tmp_path.rglob('*.jpg'))) == 3


def test_max_total_files_limit(tmp_path):
    writer = FaceCaptureWriter(
        tmp_path,
        min_interval_sec=0.0,
        max_total_files=2,
        write_image=write_dummy_image,
    )

    writer._cache_time = 0.0
    results = [writer.save(object(), score=0.7) for _ in range(4)]

    assert len([result for result in results if result]) == 2
    assert len(list(tmp_path.rglob('*.jpg'))) == 2


def test_empty_image_is_skipped(tmp_path):
    class EmptyImage:
        size = 0

    writer = FaceCaptureWriter(
        tmp_path,
        min_interval_sec=0.0,
        write_image=write_dummy_image,
    )

    assert writer.save(EmptyImage(), score=0.8) is False


def test_limit_warning_logged_once(tmp_path, caplog):
    writer = FaceCaptureWriter(
        tmp_path,
        min_interval_sec=0.0,
        max_files_per_day=1,
        write_image=write_dummy_image,
    )

    writer.save(object(), score=0.7)
    writer.save(object(), score=0.7)
    writer.save(object(), score=0.7)

    warnings = [record for record in caplog.records if '保存停止' in record.message]
    assert len(warnings) == 1

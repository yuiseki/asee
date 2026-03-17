from __future__ import annotations

import json
from pathlib import Path

from asee.curate_review_subset import (
    ReviewSample,
    materialize_subset,
    select_representative_samples,
)


def _sample(
    tmp_path: Path,
    *,
    label: str,
    name: str,
    day: str,
    hour_bucket: str,
    camera_id: int,
    captured_at: str,
) -> ReviewSample:
    source_root = tmp_path / label
    sidecar = source_root / f"{name}.json"
    image = source_root / f"{name}.jpg"
    source_root.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"jpg")
    sidecar.write_text(
        json.dumps(
            {
                "capturedAt": captured_at,
                "cameraId": camera_id,
                "roomContext": {"observedAt": "2026-03-18T06:00:00"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return ReviewSample(
        label=label,
        source_root=source_root,
        image_path=image,
        sidecar_path=sidecar,
        day=day,
        hour_bucket=hour_bucket,
        camera_id=camera_id,
        captured_at=captured_at,
        score=None,
        room_context_present=True,
    )


def test_select_representative_samples_round_robins_across_strata(tmp_path: Path) -> None:
    samples = [
        _sample(
            tmp_path,
            label="owner_raw",
            name=f"a{i}",
            day="2026-03-14",
            hour_bucket="05",
            camera_id=0,
            captured_at=f"2026-03-14T05:00:0{i}",
        )
        for i in range(3)
    ] + [
        _sample(
            tmp_path,
            label="owner_raw",
            name=f"b{i}",
            day="2026-03-15",
            hour_bucket="05",
            camera_id=4,
            captured_at=f"2026-03-15T05:10:0{i}",
        )
        for i in range(2)
    ] + [
        _sample(
            tmp_path,
            label="owner_raw",
            name="c0",
            day="2026-03-17",
            hour_bucket="06",
            camera_id=6,
            captured_at="2026-03-17T06:02:00",
        )
    ]

    selected = select_representative_samples(samples, target_count=4)

    assert len(selected) == 4
    first_three_strata = {
        (sample.day, sample.camera_id, sample.hour_bucket)
        for sample in selected[:3]
    }
    assert first_three_strata == {
        ("2026-03-14", 0, "05"),
        ("2026-03-15", 4, "05"),
        ("2026-03-17", 6, "06"),
    }


def test_materialize_subset_copies_images_and_sidecars(tmp_path: Path) -> None:
    sample = _sample(
        tmp_path,
        label="subject_false_negative",
        name="face-1",
        day="2026-03-18",
        hour_bucket="05",
        camera_id=2,
        captured_at="2026-03-18T05:55:00",
    )

    output_root = tmp_path / "out"
    summary = materialize_subset(output_root=output_root, samples=[sample])

    copied_image = output_root / "subject_false_negative" / "face-1.jpg"
    copied_sidecar = output_root / "subject_false_negative" / "face-1.json"
    assert copied_image.exists()
    assert copied_sidecar.exists()
    assert summary["sampleCount"] == 1
    assert summary["counts"] == {"subject_false_negative": 1}

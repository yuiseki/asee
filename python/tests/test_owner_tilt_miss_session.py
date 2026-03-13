from __future__ import annotations

import json
from pathlib import Path

from asee.owner_tilt_miss_session import (
    OwnerTiltMissFeature,
    OwnerTiltMissThresholds,
    copy_owner_tilt_miss_features,
    select_owner_tilt_miss_features,
)


def test_select_owner_tilt_miss_features_keeps_only_tilted_owner_only_false_negatives() -> None:
    selected = select_owner_tilt_miss_features(
        (
            OwnerTiltMissFeature(
                source_path=Path("cam0/tilted.jpg"),
                owner_score=0.52,
                owner_false_negative_similarity=0.66,
                min_face_size=96,
                blur_variance=140.0,
                detection_score=0.60,
                metadata_label="SUBJECT",
                owner_count=0,
                subject_count=1,
                people_count=1,
                roll_degrees=-12.5,
                abs_roll_degrees=12.5,
            ),
            OwnerTiltMissFeature(
                source_path=Path("cam0/upright.jpg"),
                owner_score=0.53,
                owner_false_negative_similarity=0.68,
                min_face_size=96,
                blur_variance=140.0,
                detection_score=0.61,
                metadata_label="SUBJECT",
                owner_count=0,
                subject_count=1,
                people_count=1,
                roll_degrees=5.0,
                abs_roll_degrees=5.0,
            ),
            OwnerTiltMissFeature(
                source_path=Path("cam0/mixed.jpg"),
                owner_score=0.53,
                owner_false_negative_similarity=0.68,
                min_face_size=96,
                blur_variance=140.0,
                detection_score=0.61,
                metadata_label="SUBJECT",
                owner_count=1,
                subject_count=1,
                people_count=2,
                roll_degrees=14.0,
                abs_roll_degrees=14.0,
            ),
        ),
        thresholds=OwnerTiltMissThresholds(min_abs_roll_deg=8.0),
    )

    assert [feature.source_path for feature in selected] == [Path("cam0/tilted.jpg")]


def test_copy_owner_tilt_miss_features_copies_sidecars_and_manifest(tmp_path: Path) -> None:
    source_root = tmp_path / "others"
    feature_dir = source_root / "2026" / "03" / "14" / "07" / "01"
    feature_dir.mkdir(parents=True)

    image_path = feature_dir / "tilted.jpg"
    image_path.write_bytes(b"tilted")
    image_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "capturedAt": "2026-03-14T07:01:02",
                "cameraId": 2,
                "label": "SUBJECT",
            }
        ),
        encoding="utf-8",
    )

    result = copy_owner_tilt_miss_features(
        source_root=source_root,
        selected_features=(
            OwnerTiltMissFeature(
                source_path=image_path,
                owner_score=0.52,
                owner_false_negative_similarity=0.66,
                min_face_size=96,
                blur_variance=140.0,
                detection_score=0.60,
                metadata_label="SUBJECT",
                owner_count=0,
                subject_count=1,
                people_count=1,
                roll_degrees=-12.5,
                abs_roll_degrees=12.5,
            ),
        ),
        output_root=tmp_path / "owner_tilt_miss_session",
    )

    copied_image = (
        result.output_root
        / "2026"
        / "03"
        / "14"
        / "07"
        / "01"
        / "tilted.jpg"
    )
    copied_sidecar = copied_image.with_suffix(".json")

    assert result.total_selected == 1
    assert copied_image.read_bytes() == b"tilted"
    assert json.loads(copied_sidecar.read_text(encoding="utf-8")) == {
        "capturedAt": "2026-03-14T07:01:02",
        "cameraId": 2,
        "label": "SUBJECT",
    }

    manifest = [
        json.loads(line)
        for line in result.manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    assert manifest == [
        {
            "abs_roll_degrees": 12.5,
            "detection_score": 0.6,
            "owner_count": 0,
            "owner_false_negative_similarity": 0.66,
            "owner_score": 0.52,
            "people_count": 1,
            "relative_path": "2026/03/14/07/01/tilted.jpg",
            "roll_degrees": -12.5,
            "source_path": str(image_path),
            "subject_count": 1,
        }
    ]

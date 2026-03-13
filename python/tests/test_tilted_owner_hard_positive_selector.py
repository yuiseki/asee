from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from asee.tilted_owner_hard_positive_selector import (
    TiltedHardPositiveFeature,
    copy_tilted_hard_positive_features,
    estimate_eye_line_roll_degrees,
    select_tilted_hard_positive_features,
)


def test_estimate_eye_line_roll_degrees_uses_eye_landmarks() -> None:
    detection = np.array(
        [
            0.0,
            0.0,
            100.0,
            100.0,
            10.0,
            20.0,
            30.0,
            40.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.9,
        ],
        dtype=np.float32,
    )

    assert estimate_eye_line_roll_degrees(detection) == pytest.approx(45.0, abs=0.5)


def test_select_tilted_hard_positive_features_filters_and_dedupes() -> None:
    features = (
        TiltedHardPositiveFeature(
            source_path=Path("cam0/a.jpg"),
            roll_degrees=20.0,
            abs_roll_degrees=20.0,
            detection_score=0.90,
            owner_score=0.49,
            embedding=np.array([1.0, 0.0], dtype=np.float32),
        ),
        TiltedHardPositiveFeature(
            source_path=Path("cam0/b.jpg"),
            roll_degrees=-18.0,
            abs_roll_degrees=18.0,
            detection_score=0.92,
            owner_score=0.48,
            embedding=np.array([0.999, 0.001], dtype=np.float32),
        ),
        TiltedHardPositiveFeature(
            source_path=Path("cam1/c.jpg"),
            roll_degrees=15.0,
            abs_roll_degrees=15.0,
            detection_score=0.91,
            owner_score=0.47,
            embedding=np.array([0.0, 1.0], dtype=np.float32),
        ),
        TiltedHardPositiveFeature(
            source_path=Path("cam1/d.jpg"),
            roll_degrees=8.0,
            abs_roll_degrees=8.0,
            detection_score=0.95,
            owner_score=0.70,
            embedding=np.array([0.5, 0.5], dtype=np.float32),
        ),
    )

    selected = select_tilted_hard_positive_features(
        features,
        min_abs_roll_deg=12.0,
        min_detection_score=0.5,
        min_owner_score=0.45,
        max_similarity_to_selected=0.99,
        max_selected=None,
    )

    assert [feature.source_path for feature in selected] == [
        Path("cam0/a.jpg"),
        Path("cam1/c.jpg"),
    ]


def test_copy_tilted_hard_positive_features_copies_sidecars_and_manifest(tmp_path: Path) -> None:
    source_root = tmp_path / "likely_owner_false_negative"
    feature_dir = source_root / "2026" / "03" / "14" / "07" / "01"
    feature_dir.mkdir(parents=True)

    image_path = feature_dir / "tilted.jpg"
    image_path.write_bytes(b"tilted")
    image_path.with_suffix(".json").write_text('{"label":"SUBJECT"}', encoding="utf-8")

    result = copy_tilted_hard_positive_features(
        source_root=source_root,
        selected_features=(
            TiltedHardPositiveFeature(
                source_path=image_path,
                roll_degrees=-17.5,
                abs_roll_degrees=17.5,
                detection_score=0.88,
                owner_score=0.49,
                embedding=np.array([1.0, 0.0], dtype=np.float32),
            ),
        ),
        output_root=tmp_path / "tilted_owner_hard_positive",
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
    assert json.loads(copied_sidecar.read_text(encoding="utf-8")) == {"label": "SUBJECT"}

    manifest = [
        json.loads(line)
        for line in result.manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    assert manifest == [
        {
            "abs_roll_degrees": 17.5,
            "detection_score": 0.88,
            "owner_score": 0.49,
            "relative_path": "2026/03/14/07/01/tilted.jpg",
            "roll_degrees": -17.5,
            "source_path": str(image_path),
        }
    ]

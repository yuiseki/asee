from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from asee.owner_camera_disagreement_session import (
    CaptureEvent,
    OwnerCameraDisagreementFeature,
    copy_owner_camera_disagreement_features,
    select_owner_camera_disagreement_features,
)


def test_select_owner_camera_disagreement_features_matches_nearby_owner_from_other_camera() -> None:
    subject = CaptureEvent(
        source_path=Path("others/2026/03/14/07/27/subject.jpg"),
        label="SUBJECT",
        captured_at=datetime.fromisoformat("2026-03-14T07:27:45"),
        camera_id=2,
        score=0.36,
        owner_count=0,
        subject_count=1,
        people_count=1,
    )
    owner_match = CaptureEvent(
        source_path=Path("_raw/2026/03/14/07/27/owner.jpg"),
        label="OWNER",
        captured_at=datetime.fromisoformat("2026-03-14T07:27:46"),
        camera_id=0,
        score=0.57,
        owner_count=1,
        subject_count=0,
        people_count=1,
    )
    same_camera_owner = CaptureEvent(
        source_path=Path("_raw/2026/03/14/07/27/same-camera.jpg"),
        label="OWNER",
        captured_at=datetime.fromisoformat("2026-03-14T07:27:45"),
        camera_id=2,
        score=0.59,
        owner_count=1,
        subject_count=0,
        people_count=1,
    )
    far_owner = CaptureEvent(
        source_path=Path("_raw/2026/03/14/07/30/far.jpg"),
        label="OWNER",
        captured_at=datetime.fromisoformat("2026-03-14T07:30:45"),
        camera_id=1,
        score=0.59,
        owner_count=1,
        subject_count=0,
        people_count=1,
    )

    selected = select_owner_camera_disagreement_features(
        subject_events=(subject,),
        owner_events=(owner_match, same_camera_owner, far_owner),
        window_seconds=2.0,
    )

    assert selected == (
        OwnerCameraDisagreementFeature(
            subject_event=subject,
            matched_owner_event=owner_match,
            matched_delta_seconds=1.0,
        ),
    )


def test_copy_owner_camera_disagreement_features_copies_subject_and_manifest(
    tmp_path: Path,
) -> None:
    subject_root = tmp_path / "others"
    owner_root = tmp_path / "_raw"
    subject_dir = subject_root / "2026" / "03" / "14" / "07" / "27"
    owner_dir = owner_root / "2026" / "03" / "14" / "07" / "27"
    subject_dir.mkdir(parents=True)
    owner_dir.mkdir(parents=True)

    subject_image = subject_dir / "subject.jpg"
    subject_sidecar = subject_image.with_suffix(".json")
    owner_image = owner_dir / "owner.jpg"
    subject_image.write_bytes(b"subject")
    subject_sidecar.write_text(
        json.dumps({"cameraId": 2, "capturedAt": "2026-03-14T07:27:45", "label": "SUBJECT"}),
        encoding="utf-8",
    )
    owner_image.write_bytes(b"owner")

    result = copy_owner_camera_disagreement_features(
        subject_root=subject_root,
        selected_features=(
            OwnerCameraDisagreementFeature(
                subject_event=CaptureEvent(
                    source_path=subject_image,
                    label="SUBJECT",
                    captured_at=datetime.fromisoformat("2026-03-14T07:27:45"),
                    camera_id=2,
                    score=0.36,
                    owner_count=0,
                    subject_count=1,
                    people_count=1,
                ),
                matched_owner_event=CaptureEvent(
                    source_path=owner_image,
                    label="OWNER",
                    captured_at=datetime.fromisoformat("2026-03-14T07:27:46"),
                    camera_id=0,
                    score=0.57,
                    owner_count=1,
                    subject_count=0,
                    people_count=1,
                ),
                matched_delta_seconds=1.0,
            ),
        ),
        output_root=tmp_path / "owner_camera_disagreement_session",
    )

    copied_image = (
        result.output_root
        / "2026"
        / "03"
        / "14"
        / "07"
        / "27"
        / "subject.jpg"
    )
    assert copied_image.read_bytes() == b"subject"
    assert json.loads(copied_image.with_suffix(".json").read_text(encoding="utf-8")) == {
        "cameraId": 2,
        "capturedAt": "2026-03-14T07:27:45",
        "label": "SUBJECT",
    }

    manifest = [
        json.loads(line)
        for line in result.manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    assert manifest == [
        {
            "matched_delta_seconds": 1.0,
            "matched_owner_camera_id": 0,
            "matched_owner_score": 0.57,
            "matched_owner_source_path": str(owner_image),
            "relative_path": "2026/03/14/07/27/subject.jpg",
            "source_path": str(subject_image),
            "subject_camera_id": 2,
            "subject_score": 0.36,
        }
    ]

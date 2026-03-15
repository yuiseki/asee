from __future__ import annotations

import json
from pathlib import Path

from asee.golden_review_workflow import (
    build_label_studio_config,
    build_source_key,
    prepare_golden_review_workspace,
)


def test_build_source_key_uses_stable_tail_components() -> None:
    path = Path("/tmp/faces/owner_false_negative/2026/03/15/16")
    assert build_source_key(path) == "owner_false_negative-2026-03-15-16"


def test_build_label_studio_config_contains_expected_choices() -> None:
    config = build_label_studio_config()
    assert "owner_positive" in config
    assert "guest_negative" in config
    assert "non_face_negative" in config
    assert "uncertain" in config


def test_prepare_golden_review_workspace_writes_assets_and_review_files(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "others" / "2026" / "03" / "15" / "16" / "59"
    source_root.mkdir(parents=True)
    image_path = source_root / "face-001.jpg"
    image_path.write_bytes(b"jpg")
    sidecar_path = source_root / "face-001.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "cameraId": 2,
                "label": "SUBJECT",
                "score": 0.41,
                "frameCounts": {"ownerCount": 0, "subjectCount": 1, "peopleCount": 1},
            }
        )
    )

    workspace = tmp_path / "workspace"
    summary = prepare_golden_review_workspace(
        sources=[source_root.parent],
        workspace_dir=workspace,
        copy_files=False,
        fiftyone_python=Path("/opt/fiftyone/bin/python"),
        label_studio_python=Path("/opt/label-studio/bin/python"),
    )

    assert summary.sample_count == 1
    asset_image = next((workspace / "assets").rglob("*.jpg"))
    assert asset_image.is_symlink()
    assert asset_image.resolve() == image_path
    assert asset_image.with_suffix(".json").resolve() == sidecar_path

    manifest_path = workspace / "manifest.jsonl"
    manifest_rows = [
        json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    assert manifest_rows[0]["source_key"] == "others-2026-03-15-16"
    assert manifest_rows[0]["camera_id"] == 2
    assert manifest_rows[0]["existing_label"] == "SUBJECT"

    tasks_path = workspace / "label_studio" / "tasks.json"
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    assert tasks[0]["data"]["image"].startswith("/data/local-files/?d=assets/")
    assert tasks[0]["data"]["camera_id"] == "2"
    assert tasks[0]["data"]["existing_label"] == "SUBJECT"

    config_path = workspace / "label_studio" / "config.xml"
    assert "ASEE face crop golden review" in config_path.read_text(encoding="utf-8")

    fiftyone_script = (workspace / "launch_fiftyone.sh").read_text(encoding="utf-8")
    assert "/opt/fiftyone/bin/python" in fiftyone_script
    assert 'export PYTHONPATH="/opt:${PYTHONPATH:-}"' in fiftyone_script
    assert "fo.launch_app" in fiftyone_script
    assert "auto=False" in fiftyone_script
    assert "FiftyOne URL:" in fiftyone_script

    label_studio_script = (workspace / "launch_label_studio.sh").read_text(
        encoding="utf-8"
    )
    assert "/opt/label-studio/bin/python" in label_studio_script
    assert "LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true" in label_studio_script

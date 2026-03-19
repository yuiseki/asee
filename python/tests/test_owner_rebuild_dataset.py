from __future__ import annotations

import json
from pathlib import Path

from asee.compare_owner_embedding_strategies import ReviewedSample
from asee.owner_rebuild_dataset import (
    SplitRatios,
    materialize_split_dataset,
    split_review_samples,
)


def _sample(
    tmp_path: Path,
    *,
    project_name: str,
    label: str,
    task_id: int,
    image_name: str,
) -> ReviewedSample:
    image_path = tmp_path / "images" / image_name
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"jpg")
    sidecar_path = image_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps({"label": label}), encoding="utf-8")
    return ReviewedSample(
        project_name=project_name,
        label=label,
        source_image=image_path,
        source_sidecar=sidecar_path,
        task_id=task_id,
    )


def test_split_review_samples_is_deterministic_and_preserves_all_samples(tmp_path: Path) -> None:
    samples = tuple(
        _sample(
            tmp_path,
            project_name="project-a" if index % 2 == 0 else "project-b",
            label="owner_positive" if index < 10 else "guest_negative",
            task_id=index + 1,
            image_name=f"sample-{index:02d}.jpg",
        )
        for index in range(20)
    )

    first = split_review_samples(samples, seed=1234)
    second = split_review_samples(samples, seed=1234)

    assert first == second
    assert {
        sample.source_image
        for split_samples in first.values()
        for sample in split_samples
    } == {sample.source_image for sample in samples}
    assert sum(len(split_samples) for split_samples in first.values()) == len(samples)


def test_split_review_samples_balances_each_label_across_splits(tmp_path: Path) -> None:
    samples = tuple(
        _sample(
            tmp_path,
            project_name="project-a",
            label="owner_positive" if index < 10 else "non_face_negative",
            task_id=index + 1,
            image_name=f"balanced-{index:02d}.jpg",
        )
        for index in range(20)
    )

    splits = split_review_samples(samples, seed=20260320, ratios=SplitRatios(0.6, 0.2, 0.2))

    owner_counts = {
        split_name: sum(sample.label == "owner_positive" for sample in split_samples)
        for split_name, split_samples in splits.items()
    }
    negative_counts = {
        split_name: sum(sample.label == "non_face_negative" for sample in split_samples)
        for split_name, split_samples in splits.items()
    }

    assert owner_counts == {"train": 6, "valid": 2, "test": 2}
    assert negative_counts == {"train": 6, "valid": 2, "test": 2}


def test_materialize_split_dataset_writes_symlinks_manifest_and_summary(tmp_path: Path) -> None:
    train_owner = _sample(
        tmp_path,
        project_name="project-owner",
        label="owner_positive",
        task_id=1,
        image_name="train-owner.jpg",
    )
    valid_guest = _sample(
        tmp_path,
        project_name="project-guest",
        label="guest_negative",
        task_id=2,
        image_name="valid-guest.jpg",
    )
    output_root = tmp_path / "dataset"

    summary = materialize_split_dataset(
        split_samples={
            "train": (train_owner,),
            "valid": (valid_guest,),
            "test": (),
        },
        output_root=output_root,
        project_exports={"owner": tmp_path / "owner-export.json"},
        seed=20260320,
        copy_files=False,
    )

    train_image = (
        output_root
        / "train"
        / "owner_positive"
        / "project-owner"
        / "0001-train-owner.jpg"
    )
    train_sidecar = train_image.with_suffix(".json")
    valid_image = (
        output_root
        / "valid"
        / "guest_negative"
        / "project-guest"
        / "0001-valid-guest.jpg"
    )
    manifest_rows = [
        json.loads(line)
        for line in summary.manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    written_summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))

    assert train_image.is_symlink()
    assert train_sidecar.is_symlink()
    assert valid_image.is_symlink()
    assert len(manifest_rows) == 2
    assert summary.split_counts == {
        "train": {"owner_positive": 1},
        "valid": {"guest_negative": 1},
        "test": {},
    }
    assert written_summary["split_counts"] == summary.split_counts

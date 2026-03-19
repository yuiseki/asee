from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from asee.compare_owner_embedding_strategies import ReviewedSample
from asee.owner_rebuild_dataset import materialize_split_dataset
from asee.owner_rebuild_split_experiment import (
    load_split_dataset,
    run_owner_rebuild_split_experiment,
)


class _CosineRecognizer:
    def match(self, reference: np.ndarray, embedding: np.ndarray, metric: int) -> float:
        del metric
        left = np.asarray(reference, dtype=np.float32).reshape(-1)
        right = np.asarray(embedding, dtype=np.float32).reshape(-1)
        left /= np.linalg.norm(left)
        right /= np.linalg.norm(right)
        return float(np.dot(left, right))


class _OverlayStub:
    def __init__(self) -> None:
        self._recognizer = _CosineRecognizer()

    def extract_embedding(self, frame: np.ndarray, face_box: object) -> np.ndarray:
        del face_box
        return np.asarray(frame, dtype=np.float32)


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


def test_load_split_dataset_reads_materialized_manifest(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
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
    test_non_face = _sample(
        tmp_path,
        project_name="project-non-face",
        label="non_face_negative",
        task_id=3,
        image_name="test-non-face.jpg",
    )
    materialize_split_dataset(
        split_samples={
            "train": (train_owner,),
            "valid": (valid_guest,),
            "test": (test_non_face,),
        },
        output_root=dataset_root,
        project_exports={"dummy": tmp_path / "dummy-export.json"},
        seed=20260320,
        copy_files=False,
    )

    dataset = load_split_dataset(dataset_root)

    assert [sample.source_image.name for sample in dataset.train_owner_positive] == [
        "0001-train-owner.jpg"
    ]
    assert [sample.source_image.name for sample in dataset.valid_guest_negative] == [
        "0001-valid-guest.jpg"
    ]
    assert [sample.source_image.name for sample in dataset.test_non_face_negative] == [
        "0001-test-non-face.jpg"
    ]


def test_run_owner_rebuild_split_experiment_reports_append_and_rebuild(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    train_owner_1 = _sample(
        tmp_path,
        project_name="project-owner",
        label="owner_positive",
        task_id=1,
        image_name="train-owner-1.jpg",
    )
    train_owner_2 = _sample(
        tmp_path,
        project_name="project-owner",
        label="owner_positive",
        task_id=2,
        image_name="train-owner-2.jpg",
    )
    train_guest = _sample(
        tmp_path,
        project_name="project-guest",
        label="guest_negative",
        task_id=3,
        image_name="train-guest.jpg",
    )
    train_non_face = _sample(
        tmp_path,
        project_name="project-non-face",
        label="non_face_negative",
        task_id=4,
        image_name="train-non-face.jpg",
    )
    valid_owner = _sample(
        tmp_path,
        project_name="project-owner",
        label="owner_positive",
        task_id=5,
        image_name="valid-owner.jpg",
    )
    valid_guest = _sample(
        tmp_path,
        project_name="project-guest",
        label="guest_negative",
        task_id=6,
        image_name="valid-guest.jpg",
    )
    valid_non_face = _sample(
        tmp_path,
        project_name="project-non-face",
        label="non_face_negative",
        task_id=7,
        image_name="valid-non-face.jpg",
    )
    test_owner = _sample(
        tmp_path,
        project_name="project-owner",
        label="owner_positive",
        task_id=8,
        image_name="test-owner.jpg",
    )
    test_guest = _sample(
        tmp_path,
        project_name="project-guest",
        label="guest_negative",
        task_id=9,
        image_name="test-guest.jpg",
    )
    test_non_face = _sample(
        tmp_path,
        project_name="project-non-face",
        label="non_face_negative",
        task_id=10,
        image_name="test-non-face.jpg",
    )
    materialize_split_dataset(
        split_samples={
            "train": (train_owner_1, train_owner_2, train_guest, train_non_face),
            "valid": (valid_owner, valid_guest, valid_non_face),
            "test": (test_owner, test_guest, test_non_face),
        },
        output_root=dataset_root,
        project_exports={"dummy": tmp_path / "dummy-export.json"},
        seed=20260320,
        copy_files=False,
    )

    owner_embedding_path = tmp_path / "owner_embedding.npy"
    np.save(owner_embedding_path, np.asarray([[[1.0, 0.0, 0.0]]], dtype=np.float32))
    embedding_lookup = {
        dataset_root
        / "train"
        / "owner_positive"
        / "project-owner"
        / "0001-train-owner-1.jpg": np.asarray([[0.45, 0.89, 0.0]], dtype=np.float32),
        dataset_root
        / "train"
        / "owner_positive"
        / "project-owner"
        / "0002-train-owner-2.jpg": np.asarray([[0.40, 0.92, 0.0]], dtype=np.float32),
        dataset_root
        / "train"
        / "guest_negative"
        / "project-guest"
        / "0001-train-guest.jpg": np.asarray([[-1.0, 0.0, 0.0]], dtype=np.float32),
        dataset_root
        / "train"
        / "non_face_negative"
        / "project-non-face"
        / "0001-train-non-face.jpg": np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        dataset_root
        / "valid"
        / "owner_positive"
        / "project-owner"
        / "0001-valid-owner.jpg": np.asarray([[0.42, 0.91, 0.0]], dtype=np.float32),
        dataset_root
        / "valid"
        / "guest_negative"
        / "project-guest"
        / "0001-valid-guest.jpg": np.asarray([[-1.0, 0.0, 0.0]], dtype=np.float32),
        dataset_root
        / "valid"
        / "non_face_negative"
        / "project-non-face"
        / "0001-valid-non-face.jpg": np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        dataset_root
        / "test"
        / "owner_positive"
        / "project-owner"
        / "0001-test-owner.jpg": np.asarray([[0.39, 0.92, 0.0]], dtype=np.float32),
        dataset_root
        / "test"
        / "guest_negative"
        / "project-guest"
        / "0001-test-guest.jpg": np.asarray([[-1.0, 0.0, 0.0]], dtype=np.float32),
        dataset_root
        / "test"
        / "non_face_negative"
        / "project-non-face"
        / "0001-test-non-face.jpg": np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
    }

    report = run_owner_rebuild_split_experiment(
        dataset_root=dataset_root,
        owner_embedding_path=owner_embedding_path,
        snapshot_dir=tmp_path / "snapshots",
        negative_penalties=(3.0,),
        overlay=_OverlayStub(),
        embedding_lookup=embedding_lookup,
    )

    append = report.append_results[0]

    assert report.current.evaluation.valid_owner_positive.owner_predictions == 0
    assert append.evaluation.valid_owner_positive.owner_predictions == 1
    assert append.evaluation.valid_guest_negative.owner_predictions == 0
    assert report.rebuild_all.bank_size == 2
    assert report.rebuild_greedy_results[0].bank_size >= 1
    assert report.summary_path.exists()

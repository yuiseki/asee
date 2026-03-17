from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from asee.compare_owner_embedding_strategies import (
    ReviewedSample,
    build_review_bundle,
    compare_owner_embedding_strategies,
    load_review_samples,
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


def _write_export(tmp_path: Path, name: str, labels: list[tuple[str, str]]) -> Path:
    export_path = tmp_path / name / "2026-03-16_00-00-00" / "export-json.json"
    export_path.parent.mkdir(parents=True, exist_ok=True)
    tasks = []
    for index, (label, source_image) in enumerate(labels, start=1):
        tasks.append(
            {
                "id": index,
                "meta": {
                    "source_image": source_image,
                    "source_sidecar": f"{source_image}.json",
                },
                "annotations": [
                    {
                        "result": [
                            {
                                "type": "choices",
                                "value": {"choices": [label]},
                            }
                        ]
                    }
                ],
            }
        )
    export_path.write_text(json.dumps(tasks), encoding="utf-8")
    return export_path


def test_load_review_samples_reads_label_studio_export(tmp_path: Path) -> None:
    export_path = _write_export(
        tmp_path,
        "project",
        [("owner_positive", "/tmp/owner-1.jpg"), ("non_face_negative", "/tmp/nonface-1.jpg")],
    )

    samples = load_review_samples(export_path, project_name="project-1")

    assert samples == (
        ReviewedSample(
            project_name="project-1",
            label="owner_positive",
            source_image=Path("/tmp/owner-1.jpg"),
            source_sidecar=Path("/tmp/owner-1.jpg.json"),
            task_id=1,
        ),
        ReviewedSample(
            project_name="project-1",
            label="non_face_negative",
            source_image=Path("/tmp/nonface-1.jpg"),
            source_sidecar=Path("/tmp/nonface-1.jpg.json"),
            task_id=2,
        ),
    )


def test_build_review_bundle_partitions_projects_by_role(tmp_path: Path) -> None:
    project1 = _write_export(
        tmp_path,
        "project1",
        [
            ("owner_positive", "/tmp/p1-owner.jpg"),
            ("guest_negative", "/tmp/p1-guest.jpg"),
            ("non_face_negative", "/tmp/p1-nonface.jpg"),
            ("uncertain", "/tmp/p1-uncertain.jpg"),
        ],
    )
    project2 = _write_export(
        tmp_path,
        "project2",
        [
            ("owner_positive", "/tmp/p2-owner.jpg"),
            ("non_face_negative", "/tmp/p2-nonface.jpg"),
        ],
    )
    project3 = _write_export(
        tmp_path,
        "project3",
        [("owner_positive", "/tmp/p3-owner.jpg")],
    )
    project4 = _write_export(
        tmp_path,
        "project4",
        [
            ("owner_positive", "/tmp/p4-owner.jpg"),
            ("non_face_negative", "/tmp/p4-nonface.jpg"),
        ],
    )
    project5 = _write_export(
        tmp_path,
        "project5",
        [
            ("owner_positive", "/tmp/p5-owner.jpg"),
            ("guest_negative", "/tmp/p5-guest.jpg"),
        ],
    )
    project6 = _write_export(
        tmp_path,
        "project6",
        [
            ("owner_positive", "/tmp/p6-owner.jpg"),
            ("non_face_negative", "/tmp/p6-nonface.jpg"),
        ],
    )

    bundle = build_review_bundle(
        hard_positive_export=project1,
        baseline_contacts_export=project2,
        baseline_makeup_export=project3,
        non_face_hard_negative_export=project4,
        baseline_holdout_export=project5,
        dark_room_morning_export=project6,
    )

    assert [sample.source_image.name for sample in bundle.hard_positive_glasses] == ["p1-owner.jpg"]
    assert [sample.source_image.name for sample in bundle.baseline_contacts] == ["p2-owner.jpg"]
    assert [sample.source_image.name for sample in bundle.baseline_makeup] == ["p3-owner.jpg"]
    assert [sample.source_image.name for sample in bundle.non_face_owner_positives] == [
        "p4-owner.jpg"
    ]
    assert [sample.source_image.name for sample in bundle.baseline_holdout] == ["p5-owner.jpg"]
    assert [sample.source_image.name for sample in bundle.dark_room_morning] == ["p6-owner.jpg"]
    assert [sample.source_image.name for sample in bundle.guest_negative] == [
        "p1-guest.jpg",
        "p5-guest.jpg",
    ]
    assert [sample.source_image.name for sample in bundle.non_face_negative] == [
        "p1-nonface.jpg",
        "p2-nonface.jpg",
        "p4-nonface.jpg",
        "p6-nonface.jpg",
    ]
    assert bundle.weak_non_makeup_owner_raw == ()
    assert bundle.weak_non_makeup_false_negative == ()
    assert bundle.weak_makeup_owner_raw == ()
    assert bundle.weak_makeup_false_negative == ()


def test_build_review_bundle_loads_weak_capture_datasets(tmp_path: Path) -> None:
    project1 = _write_export(
        tmp_path,
        "project1",
        [("owner_positive", "/tmp/p1-owner.jpg")],
    )
    project2 = _write_export(
        tmp_path,
        "project2",
        [("owner_positive", "/tmp/p2-owner.jpg")],
    )
    project3 = _write_export(
        tmp_path,
        "project3",
        [("owner_positive", "/tmp/p3-owner.jpg")],
    )
    project6 = _write_export(
        tmp_path,
        "project6",
        [("owner_positive", "/tmp/p6-owner.jpg")],
    )
    non_makeup_root = tmp_path / "owner_baseline_non_makeup" / "2026-03-17_10-00_to_15-59"
    makeup_root = tmp_path / "owner_baseline_makeup" / "2026-03-17_16-00_to_17-20"
    for dataset_root, image_name, label in (
        (non_makeup_root / "owner_raw" / "10" / "00", "nm-owner.jpg", "OWNER"),
        (non_makeup_root / "subject_false_negative" / "10" / "01", "nm-fn.jpg", "SUBJECT"),
        (makeup_root / "owner_raw" / "16" / "00", "m-owner.jpg", "OWNER"),
        (makeup_root / "subject_false_negative" / "16" / "01", "m-fn.jpg", "SUBJECT"),
    ):
        dataset_root.mkdir(parents=True, exist_ok=True)
        image_path = dataset_root / image_name
        sidecar_path = image_path.with_suffix(".json")
        image_path.write_bytes(b"jpg")
        sidecar_path.write_text(
            json.dumps(
                {
                    "capturedAt": "2026-03-17T10:00:00",
                    "label": label,
                    "cameraId": 4,
                }
            ),
            encoding="utf-8",
        )

    bundle = build_review_bundle(
        hard_positive_export=project1,
        baseline_contacts_export=project2,
        baseline_makeup_export=project3,
        dark_room_morning_export=project6,
        weak_baseline_non_makeup_root=non_makeup_root,
        weak_baseline_makeup_root=makeup_root,
    )

    assert [sample.source_image.name for sample in bundle.weak_non_makeup_owner_raw] == [
        "nm-owner.jpg"
    ]
    assert [sample.source_image.name for sample in bundle.weak_non_makeup_false_negative] == [
        "nm-fn.jpg"
    ]
    assert [sample.source_image.name for sample in bundle.weak_makeup_owner_raw] == [
        "m-owner.jpg"
    ]
    assert [sample.source_image.name for sample in bundle.weak_makeup_false_negative] == [
        "m-fn.jpg"
    ]
    assert [sample.source_image.name for sample in bundle.dark_room_morning] == ["p6-owner.jpg"]


def test_compare_owner_embedding_strategies_reports_current_append_and_rebuild(
    tmp_path: Path,
) -> None:
    project1 = _write_export(
        tmp_path,
        "project1",
        [
            ("owner_positive", "/tmp/hard-1.jpg"),
            ("owner_positive", "/tmp/hard-2.jpg"),
            ("guest_negative", "/tmp/guest-1.jpg"),
            ("non_face_negative", "/tmp/nonface-1.jpg"),
        ],
    )
    project2 = _write_export(
        tmp_path,
        "project2",
        [
            ("owner_positive", "/tmp/base-1.jpg"),
            ("non_face_negative", "/tmp/nonface-2.jpg"),
        ],
    )
    project3 = _write_export(
        tmp_path,
        "project3",
        [("owner_positive", "/tmp/makeup-1.jpg")],
    )
    project4 = _write_export(
        tmp_path,
        "project4",
        [
            ("owner_positive", "/tmp/nonface-owner-1.jpg"),
            ("non_face_negative", "/tmp/nonface-3.jpg"),
        ],
    )
    project5 = _write_export(
        tmp_path,
        "project5",
        [("owner_positive", "/tmp/holdout-1.jpg")],
    )
    project6 = _write_export(
        tmp_path,
        "project6",
        [
            ("owner_positive", "/tmp/dark-1.jpg"),
            ("non_face_negative", "/tmp/dark-nonface.jpg"),
        ],
    )
    non_makeup_root = tmp_path / "owner_baseline_non_makeup" / "2026-03-17_10-00_to_15-59"
    makeup_root = tmp_path / "owner_baseline_makeup" / "2026-03-17_16-00_to_17-20"
    for dataset_root, image_name in (
        (non_makeup_root / "owner_raw" / "10" / "00", "nm-owner.jpg"),
        (non_makeup_root / "subject_false_negative" / "10" / "01", "nm-fn.jpg"),
        (makeup_root / "owner_raw" / "16" / "00", "m-owner.jpg"),
        (makeup_root / "subject_false_negative" / "16" / "01", "m-fn.jpg"),
    ):
        dataset_root.mkdir(parents=True, exist_ok=True)
        image_path = dataset_root / image_name
        image_path.write_bytes(b"jpg")
        image_path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "capturedAt": "2026-03-17T10:00:00",
                    "label": "OWNER" if "owner" in image_name else "SUBJECT",
                    "cameraId": 4,
                }
            ),
            encoding="utf-8",
        )
    owner_embedding_path = tmp_path / "owner_embedding.npy"
    np.save(
        owner_embedding_path,
        np.asarray([[[1.0, 0.0, 0.0]]], dtype=np.float32),
    )
    embedding_lookup = {
        Path("/tmp/hard-1.jpg"): np.asarray([[0.45, 0.89, 0.0]], dtype=np.float32),
        Path("/tmp/hard-2.jpg"): np.asarray([[0.40, 0.92, 0.0]], dtype=np.float32),
        Path("/tmp/base-1.jpg"): np.asarray([[0.95, 0.05, 0.0]], dtype=np.float32),
        Path("/tmp/makeup-1.jpg"): np.asarray([[0.93, 0.08, 0.0]], dtype=np.float32),
        Path("/tmp/nonface-owner-1.jpg"): np.asarray([[0.91, 0.09, 0.0]], dtype=np.float32),
        Path("/tmp/holdout-1.jpg"): np.asarray([[0.94, 0.06, 0.0]], dtype=np.float32),
        Path("/tmp/dark-1.jpg"): np.asarray([[0.36, 0.93, 0.0]], dtype=np.float32),
        non_makeup_root / "owner_raw" / "10" / "00" / "nm-owner.jpg": np.asarray(
            [[0.96, 0.04, 0.0]], dtype=np.float32
        ),
        non_makeup_root / "subject_false_negative" / "10" / "01" / "nm-fn.jpg": np.asarray(
            [[0.35, 0.93, 0.0]], dtype=np.float32
        ),
        makeup_root / "owner_raw" / "16" / "00" / "m-owner.jpg": np.asarray(
            [[0.97, 0.03, 0.0]], dtype=np.float32
        ),
        makeup_root / "subject_false_negative" / "16" / "01" / "m-fn.jpg": np.asarray(
            [[0.32, 0.95, 0.0]], dtype=np.float32
        ),
        Path("/tmp/guest-1.jpg"): np.asarray([[-1.0, 0.0, 0.0]], dtype=np.float32),
        Path("/tmp/nonface-1.jpg"): np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        Path("/tmp/nonface-2.jpg"): np.asarray([[0.0, -1.0, 0.0]], dtype=np.float32),
        Path("/tmp/nonface-3.jpg"): np.asarray([[0.0, 0.2, 0.98]], dtype=np.float32),
        Path("/tmp/dark-nonface.jpg"): np.asarray([[0.0, -1.0, 0.0]], dtype=np.float32),
    }

    report = compare_owner_embedding_strategies(
        owner_embedding_path=owner_embedding_path,
        hard_positive_export=project1,
        baseline_contacts_export=project2,
        baseline_makeup_export=project3,
        non_face_hard_negative_export=project4,
        baseline_holdout_export=project5,
        dark_room_morning_export=project6,
        weak_baseline_non_makeup_root=non_makeup_root,
        weak_baseline_makeup_root=makeup_root,
        snapshot_dir=tmp_path / "snapshots",
        overlay=_OverlayStub(),
        embedding_lookup=embedding_lookup,
    )

    assert report.current.hard_positive_glasses.owner_predictions == 0
    assert report.append.hard_positive_glasses.owner_predictions == 2
    assert report.rebuild.hard_positive_glasses.owner_predictions == 2
    assert report.current.baseline_contacts.owner_predictions == 1
    assert report.current.non_face_owner_positives.owner_predictions == 1
    assert report.current.baseline_holdout.owner_predictions == 1
    assert report.current.dark_room_morning.owner_predictions == 0
    assert report.current.weak_non_makeup_owner_raw.owner_predictions == 1
    assert report.current.weak_makeup_owner_raw.owner_predictions == 1
    assert report.current.weak_non_makeup_false_negative.owner_predictions == 0
    assert report.current.weak_makeup_false_negative.owner_predictions == 0
    assert report.append.guest_negative.owner_predictions == 0
    assert report.rebuild.non_face_negative.owner_predictions == 0
    assert report.append.weak_non_makeup_false_negative.owner_predictions == 1
    assert report.append.weak_makeup_false_negative.owner_predictions == 1
    assert report.append.added_embeddings == 1
    assert report.rebuild.bank_size == 4
    assert report.append.candidate_embedding_path is not None
    assert report.rebuild.candidate_embedding_path is not None

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

    bundle = build_review_bundle(
        hard_positive_export=project1,
        baseline_contacts_export=project2,
        baseline_makeup_export=project3,
    )

    assert [sample.source_image.name for sample in bundle.hard_positive_glasses] == ["p1-owner.jpg"]
    assert [sample.source_image.name for sample in bundle.baseline_contacts] == ["p2-owner.jpg"]
    assert [sample.source_image.name for sample in bundle.baseline_makeup] == ["p3-owner.jpg"]
    assert [sample.source_image.name for sample in bundle.guest_negative] == ["p1-guest.jpg"]
    assert [sample.source_image.name for sample in bundle.non_face_negative] == [
        "p1-nonface.jpg",
        "p2-nonface.jpg",
    ]


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
        Path("/tmp/guest-1.jpg"): np.asarray([[-1.0, 0.0, 0.0]], dtype=np.float32),
        Path("/tmp/nonface-1.jpg"): np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        Path("/tmp/nonface-2.jpg"): np.asarray([[0.0, -1.0, 0.0]], dtype=np.float32),
    }

    report = compare_owner_embedding_strategies(
        owner_embedding_path=owner_embedding_path,
        hard_positive_export=project1,
        baseline_contacts_export=project2,
        baseline_makeup_export=project3,
        snapshot_dir=tmp_path / "snapshots",
        overlay=_OverlayStub(),
        embedding_lookup=embedding_lookup,
    )

    assert report.current.hard_positive_glasses.owner_predictions == 0
    assert report.append.hard_positive_glasses.owner_predictions == 2
    assert report.rebuild.hard_positive_glasses.owner_predictions == 2
    assert report.current.baseline_contacts.owner_predictions == 1
    assert report.append.guest_negative.owner_predictions == 0
    assert report.rebuild.non_face_negative.owner_predictions == 0
    assert report.append.added_embeddings == 1
    assert report.rebuild.bank_size == 4
    assert report.append.candidate_embedding_path is not None
    assert report.rebuild.candidate_embedding_path is not None

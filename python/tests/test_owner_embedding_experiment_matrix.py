from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from asee.compare_owner_embedding_strategies import build_review_bundle
from asee.owner_embedding_experiment_matrix import (
    ExperimentSourceGroup,
    ExperimentStrategy,
    build_default_source_groups,
    run_owner_embedding_experiment_matrix,
)


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


def test_build_default_source_groups_generates_all_non_empty_project_combinations(
    tmp_path: Path,
) -> None:
    project1 = _write_export(tmp_path, "project1", [("owner_positive", "/tmp/p1-owner.jpg")])
    project2 = _write_export(tmp_path, "project2", [("owner_positive", "/tmp/p2-owner.jpg")])
    project3 = _write_export(tmp_path, "project3", [("owner_positive", "/tmp/p3-owner.jpg")])
    bundle = build_review_bundle(
        hard_positive_export=project1,
        baseline_contacts_export=project2,
        baseline_makeup_export=project3,
    )

    groups = build_default_source_groups(bundle)

    assert [group.key for group in groups] == [
        "hard_positive_glasses",
        "baseline_contacts",
        "baseline_makeup",
        "hard_positive_glasses+baseline_contacts",
        "hard_positive_glasses+baseline_makeup",
        "baseline_contacts+baseline_makeup",
        "hard_positive_glasses+baseline_contacts+baseline_makeup",
    ]


def test_build_default_source_groups_includes_weak_false_negative_sets(
    tmp_path: Path,
) -> None:
    project1 = _write_export(tmp_path, "project1", [("owner_positive", "/tmp/p1-owner.jpg")])
    project2 = _write_export(tmp_path, "project2", [("owner_positive", "/tmp/p2-owner.jpg")])
    project3 = _write_export(tmp_path, "project3", [("owner_positive", "/tmp/p3-owner.jpg")])
    non_makeup_root = tmp_path / "owner_baseline_non_makeup" / "2026-03-17_10-00_to_15-59"
    makeup_root = tmp_path / "owner_baseline_makeup" / "2026-03-17_16-00_to_17-20"
    for dataset_root, image_name in (
        (non_makeup_root / "subject_false_negative" / "10" / "01", "nm-fn.jpg"),
        (makeup_root / "subject_false_negative" / "16" / "01", "m-fn.jpg"),
    ):
        dataset_root.mkdir(parents=True, exist_ok=True)
        image_path = dataset_root / image_name
        image_path.write_bytes(b"jpg")
        image_path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "capturedAt": "2026-03-17T10:00:00",
                    "label": "SUBJECT",
                    "cameraId": 4,
                }
            ),
            encoding="utf-8",
        )
    bundle = build_review_bundle(
        hard_positive_export=project1,
        baseline_contacts_export=project2,
        baseline_makeup_export=project3,
        weak_baseline_non_makeup_root=non_makeup_root,
        weak_baseline_makeup_root=makeup_root,
    )

    groups = build_default_source_groups(bundle)

    keys = [group.key for group in groups]
    assert "weak_non_makeup_false_negative" in keys
    assert "weak_makeup_false_negative" in keys


def test_run_owner_embedding_experiment_matrix_reports_multiple_strategies(
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
    }

    report = run_owner_embedding_experiment_matrix(
        owner_embedding_path=owner_embedding_path,
        hard_positive_export=project1,
        baseline_contacts_export=project2,
        baseline_makeup_export=project3,
        weak_baseline_non_makeup_root=non_makeup_root,
        weak_baseline_makeup_root=makeup_root,
        snapshot_dir=tmp_path / "snapshots",
        source_groups=(
            ExperimentSourceGroup(
                key="hard_positive_glasses",
                samples=build_review_bundle(
                    hard_positive_export=project1,
                    baseline_contacts_export=project2,
                    baseline_makeup_export=project3,
                ).hard_positive_glasses,
            ),
            ExperimentSourceGroup(
                key="all_owner_positive",
                samples=build_review_bundle(
                    hard_positive_export=project1,
                    baseline_contacts_export=project2,
                    baseline_makeup_export=project3,
                    weak_baseline_non_makeup_root=non_makeup_root,
                    weak_baseline_makeup_root=makeup_root,
                ).rebuild_sources,
            ),
            ExperimentSourceGroup(
                key="weak_makeup_false_negative",
                samples=build_review_bundle(
                    hard_positive_export=project1,
                    baseline_contacts_export=project2,
                    baseline_makeup_export=project3,
                    weak_baseline_non_makeup_root=non_makeup_root,
                    weak_baseline_makeup_root=makeup_root,
                ).weak_makeup_false_negative,
            ),
        ),
        strategies=(
            ExperimentStrategy(key="append_greedy", mode="append_greedy", negative_penalty=3.0),
            ExperimentStrategy(key="rebuild", mode="rebuild"),
        ),
        embedding_lookup=embedding_lookup,
    )

    result_keys = {(result.source_group_key, result.strategy_key) for result in report.results}
    assert result_keys == {
        ("hard_positive_glasses", "append_greedy"),
        ("hard_positive_glasses", "rebuild"),
        ("all_owner_positive", "append_greedy"),
        ("all_owner_positive", "rebuild"),
        ("weak_makeup_false_negative", "append_greedy"),
        ("weak_makeup_false_negative", "rebuild"),
    }
    assert report.current.hard_positive_glasses.owner_predictions == 0
    assert report.current.weak_non_makeup_owner_raw.owner_predictions == 1
    assert report.current.weak_makeup_false_negative.owner_predictions == 0
    append_result = next(
        result
        for result in report.results
        if result.source_group_key == "hard_positive_glasses"
        and result.strategy_key == "append_greedy"
    )
    rebuild_result = next(
        result
        for result in report.results
        if result.source_group_key == "all_owner_positive"
        and result.strategy_key == "rebuild"
    )
    assert append_result.hard_positive_gain == 2
    assert append_result.negative_all_delta == 0
    assert rebuild_result.bank_size == 4
    assert rebuild_result.negative_all_delta >= 0
    assert report.summary_json_path.exists()

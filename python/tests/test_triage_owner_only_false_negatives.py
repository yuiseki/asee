from __future__ import annotations

import json
from pathlib import Path

from asee.triage_owner_only_false_negatives import (
    OwnerOnlyCandidateFeature,
    OwnerOnlyTriageThresholds,
    build_arg_parser,
    classify_owner_only_candidate,
    triage_owner_only_candidates,
)


def test_build_arg_parser_uses_numeric_defaults() -> None:
    args = build_arg_parser().parse_args([])

    assert isinstance(args.min_face_size, int)
    assert isinstance(args.min_blur_variance, float)
    assert isinstance(args.min_detection_score, float)
    assert isinstance(args.min_owner_score, float)
    assert isinstance(args.min_owner_false_negative_similarity, float)


def test_classify_owner_only_candidate_marks_likely_owner_false_negative() -> None:
    feature = OwnerOnlyCandidateFeature(
        source_path=Path("cam0/face.jpg"),
        owner_score=0.47,
        owner_false_negative_similarity=0.60,
        min_face_size=96,
        blur_variance=120.0,
        detection_score=0.52,
        metadata_label="SUBJECT",
        owner_count=0,
        subject_count=1,
        people_count=1,
    )

    assert (
        classify_owner_only_candidate(
            feature,
            thresholds=OwnerOnlyTriageThresholds(),
        )
        == "likely_owner_false_negative"
    )


def test_classify_owner_only_candidate_marks_low_quality() -> None:
    feature = OwnerOnlyCandidateFeature(
        source_path=Path("cam0/face.jpg"),
        owner_score=0.55,
        owner_false_negative_similarity=0.70,
        min_face_size=64,
        blur_variance=30.0,
        detection_score=0.30,
        metadata_label="SUBJECT",
        owner_count=0,
        subject_count=1,
        people_count=1,
    )

    assert (
        classify_owner_only_candidate(
            feature,
            thresholds=OwnerOnlyTriageThresholds(),
        )
        == "low_quality"
    )


def test_classify_owner_only_candidate_marks_uncertain_when_frame_counts_are_not_owner_only(
) -> None:
    feature = OwnerOnlyCandidateFeature(
        source_path=Path("cam0/face.jpg"),
        owner_score=0.55,
        owner_false_negative_similarity=0.70,
        min_face_size=96,
        blur_variance=120.0,
        detection_score=0.55,
        metadata_label="SUBJECT",
        owner_count=1,
        subject_count=1,
        people_count=2,
    )

    assert (
        classify_owner_only_candidate(
            feature,
            thresholds=OwnerOnlyTriageThresholds(),
        )
        == "uncertain"
    )


def test_triage_owner_only_candidates_copies_files_and_manifest(tmp_path: Path) -> None:
    source_root = tmp_path / "others"
    bucket_dir = source_root / "2026" / "03" / "14" / "07" / "01"
    bucket_dir.mkdir(parents=True)

    likely_path = bucket_dir / "likely.jpg"
    low_quality_path = bucket_dir / "low.jpg"
    uncertain_path = bucket_dir / "uncertain.jpg"
    likely_path.write_bytes(b"likely")
    low_quality_path.write_bytes(b"low")
    uncertain_path.write_bytes(b"uncertain")
    likely_path.with_suffix(".json").write_text('{"label":"SUBJECT"}', encoding="utf-8")

    output_root = tmp_path / "triaged"
    result = triage_owner_only_candidates(
        source_root=source_root,
        features=(
            OwnerOnlyCandidateFeature(
                source_path=likely_path,
                owner_score=0.47,
                owner_false_negative_similarity=0.60,
                min_face_size=96,
                blur_variance=120.0,
                detection_score=0.52,
                metadata_label="SUBJECT",
                owner_count=0,
                subject_count=1,
                people_count=1,
            ),
            OwnerOnlyCandidateFeature(
                source_path=low_quality_path,
                owner_score=0.47,
                owner_false_negative_similarity=0.60,
                min_face_size=64,
                blur_variance=20.0,
                detection_score=0.30,
                metadata_label="SUBJECT",
                owner_count=0,
                subject_count=1,
                people_count=1,
            ),
            OwnerOnlyCandidateFeature(
                source_path=uncertain_path,
                owner_score=0.44,
                owner_false_negative_similarity=0.49,
                min_face_size=96,
                blur_variance=120.0,
                detection_score=0.52,
                metadata_label="SUBJECT",
                owner_count=0,
                subject_count=1,
                people_count=1,
            ),
        ),
        output_root=output_root,
        thresholds=OwnerOnlyTriageThresholds(),
    )

    assert result.total_files == 3
    assert result.counts == {
        "likely_owner_false_negative": 1,
        "low_quality": 1,
        "uncertain": 1,
    }
    assert (
        output_root
        / "likely_owner_false_negative"
        / "2026"
        / "03"
        / "14"
        / "07"
        / "01"
        / "likely.jpg"
    ).read_bytes() == b"likely"
    assert json.loads(
        (
            output_root
            / "likely_owner_false_negative"
            / "2026"
            / "03"
            / "14"
            / "07"
            / "01"
            / "likely.json"
        ).read_text(encoding="utf-8")
    ) == {"label": "SUBJECT"}

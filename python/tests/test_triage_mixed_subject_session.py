from __future__ import annotations

import json
from pathlib import Path

from asee.triage_mixed_subject_session import (
    SessionTriageThresholds,
    SubjectSessionFeature,
    build_arg_parser,
    classify_subject_session_feature,
    triage_subject_session_features,
)


def test_classify_subject_session_feature_assigns_likely_guest_negative() -> None:
    feature = SubjectSessionFeature(
        source_path=Path("cam0/face.jpg"),
        owner_score=0.40,
        owner_false_negative_similarity=0.50,
    )

    assert (
        classify_subject_session_feature(
            feature,
            thresholds=SessionTriageThresholds(),
        )
        == "likely_guest_negative"
    )


def test_build_arg_parser_uses_numeric_default_thresholds() -> None:
    args = build_arg_parser().parse_args([])

    assert isinstance(args.max_owner_score_for_likely_guest, float)
    assert isinstance(args.max_owner_fn_similarity_for_likely_guest, float)
    assert isinstance(args.min_owner_score_for_likely_owner_false_negative, float)
    assert isinstance(
        args.min_owner_fn_similarity_for_likely_owner_false_negative,
        float,
    )


def test_classify_subject_session_feature_assigns_likely_owner_false_negative() -> None:
    feature = SubjectSessionFeature(
        source_path=Path("cam0/face.jpg"),
        owner_score=0.53,
        owner_false_negative_similarity=0.81,
    )

    assert (
        classify_subject_session_feature(
            feature,
            thresholds=SessionTriageThresholds(),
        )
        == "likely_owner_false_negative"
    )


def test_classify_subject_session_feature_assigns_uncertain_for_overlap_band() -> None:
    feature = SubjectSessionFeature(
        source_path=Path("cam0/face.jpg"),
        owner_score=0.49,
        owner_false_negative_similarity=0.70,
    )

    assert (
        classify_subject_session_feature(
            feature,
            thresholds=SessionTriageThresholds(),
        )
        == "uncertain"
    )


def test_triage_subject_session_features_copies_files_and_writes_manifest(tmp_path: Path) -> None:
    source_root = tmp_path / "mixed-session"
    feature_dir = source_root / "2026" / "03" / "13" / "19" / "27"
    feature_dir.mkdir(parents=True)

    guest_path = feature_dir / "guest.jpg"
    owner_path = feature_dir / "owner.jpg"
    unsure_path = feature_dir / "unsure.jpg"
    guest_path.write_bytes(b"guest")
    owner_path.write_bytes(b"owner")
    unsure_path.write_bytes(b"unsure")

    output_root = tmp_path / "triaged"
    result = triage_subject_session_features(
        session_root=source_root,
        features=(
            SubjectSessionFeature(
                source_path=guest_path,
                owner_score=0.40,
                owner_false_negative_similarity=0.50,
            ),
            SubjectSessionFeature(
                source_path=owner_path,
                owner_score=0.53,
                owner_false_negative_similarity=0.81,
            ),
            SubjectSessionFeature(
                source_path=unsure_path,
                owner_score=0.49,
                owner_false_negative_similarity=0.70,
            ),
        ),
        output_root=output_root,
        thresholds=SessionTriageThresholds(),
    )

    assert result.total_files == 3
    assert result.counts == {
        "likely_guest_negative": 1,
        "likely_owner_false_negative": 1,
        "uncertain": 1,
    }

    assert (
        output_root
        / "likely_guest_negative"
        / "2026"
        / "03"
        / "13"
        / "19"
        / "27"
        / "guest.jpg"
    ).read_bytes() == b"guest"
    assert (
        output_root
        / "likely_owner_false_negative"
        / "2026"
        / "03"
        / "13"
        / "19"
        / "27"
        / "owner.jpg"
    ).read_bytes() == b"owner"
    assert (
        output_root
        / "uncertain"
        / "2026"
        / "03"
        / "13"
        / "19"
        / "27"
        / "unsure.jpg"
    ).read_bytes() == b"unsure"

    manifest_records = [
        json.loads(line)
        for line in result.manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["bucket"] for record in manifest_records] == [
        "likely_guest_negative",
        "likely_owner_false_negative",
        "uncertain",
    ]

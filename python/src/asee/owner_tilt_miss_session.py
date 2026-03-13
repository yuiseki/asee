"""Collect owner-only SUBJECT tilt misses into a dedicated validation bucket."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from shutil import copy2

import numpy as np

from .enroll_owner import DEFAULT_OWNER_EMBED_PATH
from .overlay import GodModeOverlay
from .retrain_owner_embedding import DEFAULT_FALSE_NEGATIVE_DIR, read_face_crop
from .tilted_owner_hard_positive_selector import detect_primary_face_roll_degrees
from .triage_owner_only_false_negatives import (
    DEFAULT_MIN_BLUR_VARIANCE,
    DEFAULT_MIN_DETECTION_SCORE,
    DEFAULT_MIN_FACE_SIZE,
    DEFAULT_MIN_OWNER_FALSE_NEGATIVE_SIMILARITY,
    DEFAULT_MIN_OWNER_SCORE,
    OwnerOnlyCandidateFeature,
    OwnerOnlyTriageThresholds,
    build_owner_only_candidate_features,
    classify_owner_only_candidate,
)

DEFAULT_SOURCE_DIR = Path(
    "/home/yuiseki/Workspaces/private/datasets/faces/others/2026/03/14"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/yuiseki/Workspaces/private/datasets/faces/owner_tilt_miss_session"
)
DEFAULT_MIN_ABS_ROLL_DEG = 8.0
type FrameArray = np.ndarray


@dataclass(frozen=True, slots=True)
class OwnerTiltMissThresholds:
    owner_only_thresholds: OwnerOnlyTriageThresholds = field(
        default_factory=OwnerOnlyTriageThresholds
    )
    min_abs_roll_deg: float = DEFAULT_MIN_ABS_ROLL_DEG


@dataclass(frozen=True, slots=True)
class OwnerTiltMissFeature:
    source_path: Path
    owner_score: float
    owner_false_negative_similarity: float
    min_face_size: int
    blur_variance: float
    detection_score: float
    metadata_label: str
    owner_count: int | None
    subject_count: int | None
    people_count: int | None
    roll_degrees: float
    abs_roll_degrees: float


@dataclass(frozen=True, slots=True)
class OwnerTiltMissResult:
    output_root: Path
    manifest_path: Path
    total_selected: int


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect owner-only SUBJECT tilt misses into a dedicated validation bucket."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument(
        "--owner-false-negative-dir",
        type=Path,
        default=DEFAULT_FALSE_NEGATIVE_DIR,
    )
    parser.add_argument(
        "--owner-embedding-path",
        type=Path,
        default=DEFAULT_OWNER_EMBED_PATH,
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--min-face-size", type=int, default=DEFAULT_MIN_FACE_SIZE)
    parser.add_argument(
        "--min-blur-variance", type=float, default=DEFAULT_MIN_BLUR_VARIANCE
    )
    parser.add_argument(
        "--min-detection-score", type=float, default=DEFAULT_MIN_DETECTION_SCORE
    )
    parser.add_argument("--min-owner-score", type=float, default=DEFAULT_MIN_OWNER_SCORE)
    parser.add_argument(
        "--min-owner-false-negative-similarity",
        type=float,
        default=DEFAULT_MIN_OWNER_FALSE_NEGATIVE_SIMILARITY,
    )
    parser.add_argument(
        "--min-abs-roll-deg",
        type=float,
        default=DEFAULT_MIN_ABS_ROLL_DEG,
    )
    return parser


def select_owner_tilt_miss_features(
    features: Sequence[OwnerTiltMissFeature],
    *,
    thresholds: OwnerTiltMissThresholds,
) -> tuple[OwnerTiltMissFeature, ...]:
    selected = [
        feature
        for feature in features
        if classify_owner_only_candidate(
            _as_owner_only_candidate(feature),
            thresholds=thresholds.owner_only_thresholds,
        )
        == "likely_owner_false_negative"
        and feature.abs_roll_degrees >= thresholds.min_abs_roll_deg
    ]
    selected.sort(
        key=lambda feature: (
            -feature.abs_roll_degrees,
            -feature.owner_score,
            str(feature.source_path),
        )
    )
    return tuple(selected)


def copy_owner_tilt_miss_features(
    *,
    source_root: Path,
    selected_features: Sequence[OwnerTiltMissFeature],
    output_root: Path,
    copy_file: Callable[[Path, Path], object] | None = None,
) -> OwnerTiltMissResult:
    writer = copy_file or copy2
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for feature in selected_features:
            relative_path = feature.source_path.relative_to(source_root)
            destination = output_root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            writer(feature.source_path, destination)
            sidecar = feature.source_path.with_suffix(".json")
            if sidecar.exists():
                writer(sidecar, destination.with_suffix(".json"))
            manifest.write(
                json.dumps(
                    {
                        "source_path": str(feature.source_path),
                        "relative_path": str(relative_path),
                        "owner_score": feature.owner_score,
                        "owner_false_negative_similarity": feature.owner_false_negative_similarity,
                        "detection_score": feature.detection_score,
                        "roll_degrees": feature.roll_degrees,
                        "abs_roll_degrees": feature.abs_roll_degrees,
                        "owner_count": feature.owner_count,
                        "subject_count": feature.subject_count,
                        "people_count": feature.people_count,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            manifest.write("\n")
    return OwnerTiltMissResult(
        output_root=output_root,
        manifest_path=manifest_path,
        total_selected=len(selected_features),
    )


def build_owner_tilt_miss_features(
    *,
    source_dir: Path,
    owner_false_negative_dir: Path,
    owner_embedding_path: Path,
    owner_only_thresholds: OwnerOnlyTriageThresholds,
    read_image: Callable[[Path], FrameArray | None] = read_face_crop,
) -> tuple[OwnerTiltMissFeature, ...]:
    owner_only_features = build_owner_only_candidate_features(
        source_dir=source_dir,
        owner_false_negative_dir=owner_false_negative_dir,
        owner_embedding_path=owner_embedding_path,
        read_image=read_image,
    )
    detector_overlay = GodModeOverlay(width=1280, height=720)
    detector = detector_overlay._detector
    if detector is None:
        raise RuntimeError("overlay detector unavailable")
    collected: list[OwnerTiltMissFeature] = []
    for feature in owner_only_features:
        if (
            classify_owner_only_candidate(
                feature,
                thresholds=owner_only_thresholds,
            )
            != "likely_owner_false_negative"
        ):
            continue
        frame = read_image(feature.source_path)
        if frame is None:
            continue
        roll_degrees, _ = detect_primary_face_roll_degrees(frame=frame, detector=detector)
        if roll_degrees is None:
            continue
        collected.append(
            OwnerTiltMissFeature(
                source_path=feature.source_path,
                owner_score=feature.owner_score,
                owner_false_negative_similarity=feature.owner_false_negative_similarity,
                min_face_size=feature.min_face_size,
                blur_variance=feature.blur_variance,
                detection_score=feature.detection_score,
                metadata_label=feature.metadata_label,
                owner_count=feature.owner_count,
                subject_count=feature.subject_count,
                people_count=feature.people_count,
                roll_degrees=float(roll_degrees),
                abs_roll_degrees=abs(float(roll_degrees)),
            )
        )
    return tuple(collected)


def _as_owner_only_candidate(feature: OwnerTiltMissFeature) -> OwnerOnlyCandidateFeature:
    return OwnerOnlyCandidateFeature(
        source_path=feature.source_path,
        owner_score=feature.owner_score,
        owner_false_negative_similarity=feature.owner_false_negative_similarity,
        min_face_size=feature.min_face_size,
        blur_variance=feature.blur_variance,
        detection_score=feature.detection_score,
        metadata_label=feature.metadata_label,
        owner_count=feature.owner_count,
        subject_count=feature.subject_count,
        people_count=feature.people_count,
    )


def default_output_root_for_source(source: Path) -> Path:
    if len(source.parts) >= 3 and all(part.isdigit() for part in source.parts[-3:]):
        return DEFAULT_OUTPUT_ROOT / "-".join(source.parts[-3:])
    return DEFAULT_OUTPUT_ROOT / source.name


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    source = Path(args.source)
    owner_only_thresholds = OwnerOnlyTriageThresholds(
        min_face_size=int(args.min_face_size),
        min_blur_variance=float(args.min_blur_variance),
        min_detection_score=float(args.min_detection_score),
        min_owner_score=float(args.min_owner_score),
        min_owner_false_negative_similarity=float(
            args.min_owner_false_negative_similarity
        ),
    )
    thresholds = OwnerTiltMissThresholds(
        owner_only_thresholds=owner_only_thresholds,
        min_abs_roll_deg=float(args.min_abs_roll_deg),
    )
    output_root = (
        Path(args.output_root)
        if args.output_root is not None
        else default_output_root_for_source(source)
    )
    features = build_owner_tilt_miss_features(
        source_dir=source,
        owner_false_negative_dir=Path(args.owner_false_negative_dir),
        owner_embedding_path=Path(args.owner_embedding_path),
        owner_only_thresholds=owner_only_thresholds,
    )
    selected = select_owner_tilt_miss_features(features, thresholds=thresholds)
    result = copy_owner_tilt_miss_features(
        source_root=source,
        selected_features=selected,
        output_root=output_root,
    )
    print(f"output_root={result.output_root}")
    print(f"manifest={result.manifest_path}")
    print(f"total_selected={result.total_selected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

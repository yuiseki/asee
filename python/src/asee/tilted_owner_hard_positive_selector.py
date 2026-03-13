"""Select tilted owner hard positives for append-only owner embedding expansion."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import atan2, degrees
from pathlib import Path
from shutil import copy2
from typing import Any, Protocol, cast

import cv2
import numpy as np

from .detection_runtime import set_detector_input_size
from .enroll_owner import DEFAULT_OWNER_EMBED_PATH
from .overlay import GodModeOverlay
from .owner_policy import classify_owner_embedding
from .retrain_owner_embedding import extract_crop_embedding, iter_image_paths, read_face_crop

DEFAULT_SOURCE_DIR = Path(
    "/home/yuiseki/Workspaces/private/datasets/faces/others_owner_only_triaged/2026-03-14/likely_owner_false_negative"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/yuiseki/Workspaces/private/datasets/faces/tilted_owner_hard_positive_candidates"
)

type EmbeddingArray = np.ndarray
type FrameArray = np.ndarray


class DetectorLike(Protocol):
    def detect(self, frame: FrameArray) -> tuple[object | None, Sequence[Sequence[float]] | None]: ...


@dataclass(frozen=True, slots=True)
class TiltedHardPositiveFeature:
    source_path: Path
    roll_degrees: float
    abs_roll_degrees: float
    detection_score: float
    owner_score: float
    embedding: EmbeddingArray


@dataclass(frozen=True, slots=True)
class TiltedHardPositiveSelectionResult:
    output_root: Path
    manifest_path: Path
    total_selected: int


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select tilted owner hard positives from a likely_owner_false_negative bucket "
            "for append-only owner embedding experiments."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--owner-embedding-path", type=Path, default=DEFAULT_OWNER_EMBED_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--min-abs-roll-deg", type=float, default=12.0)
    parser.add_argument("--min-detection-score", type=float, default=0.45)
    parser.add_argument("--min-owner-score", type=float, default=0.43)
    parser.add_argument("--max-similarity-to-selected", type=float, default=0.995)
    parser.add_argument("--max-selected", type=int, default=24)
    return parser


def estimate_eye_line_roll_degrees(raw_detection: Sequence[float]) -> float | None:
    if len(raw_detection) < 8:
        return None
    eye_a_x, eye_a_y = float(raw_detection[4]), float(raw_detection[5])
    eye_b_x, eye_b_y = float(raw_detection[6]), float(raw_detection[7])
    delta_x = eye_b_x - eye_a_x
    delta_y = eye_b_y - eye_a_y
    if abs(delta_x) < 1e-6 and abs(delta_y) < 1e-6:
        return 0.0
    return float(degrees(atan2(delta_y, delta_x)))


def cosine_similarity(left: EmbeddingArray, right: EmbeddingArray) -> float:
    left_flat = np.asarray(left, dtype=np.float32).reshape(-1)
    right_flat = np.asarray(right, dtype=np.float32).reshape(-1)
    left_norm = float(np.linalg.norm(left_flat))
    right_norm = float(np.linalg.norm(right_flat))
    if left_norm <= 1e-6 or right_norm <= 1e-6:
        return 0.0
    return float(np.dot(left_flat, right_flat) / (left_norm * right_norm))


def select_tilted_hard_positive_features(
    features: Sequence[TiltedHardPositiveFeature],
    *,
    min_abs_roll_deg: float,
    min_detection_score: float,
    min_owner_score: float,
    max_similarity_to_selected: float,
    max_selected: int | None,
) -> tuple[TiltedHardPositiveFeature, ...]:
    selected: list[TiltedHardPositiveFeature] = []
    ordered = sorted(
        (
            feature
            for feature in features
            if feature.abs_roll_degrees >= min_abs_roll_deg
            and feature.detection_score >= min_detection_score
            and feature.owner_score >= min_owner_score
        ),
        key=lambda feature: (
            -feature.abs_roll_degrees,
            -feature.owner_score,
            str(feature.source_path),
        ),
    )
    for feature in ordered:
        if max_selected is not None and len(selected) >= max_selected:
            break
        if any(
            cosine_similarity(feature.embedding, existing.embedding) >= max_similarity_to_selected
            for existing in selected
        ):
            continue
        selected.append(feature)
    return tuple(selected)


def copy_tilted_hard_positive_features(
    *,
    source_root: Path,
    selected_features: Sequence[TiltedHardPositiveFeature],
    output_root: Path,
    copy_file: Callable[[Path, Path], object] | None = None,
) -> TiltedHardPositiveSelectionResult:
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
                        "roll_degrees": feature.roll_degrees,
                        "abs_roll_degrees": feature.abs_roll_degrees,
                        "detection_score": feature.detection_score,
                        "owner_score": feature.owner_score,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            manifest.write("\n")
    return TiltedHardPositiveSelectionResult(
        output_root=output_root,
        manifest_path=manifest_path,
        total_selected=len(selected_features),
    )


def build_tilted_hard_positive_features(
    *,
    source_dir: Path,
    owner_embedding_path: Path,
    read_image: Callable[[Path], FrameArray | None] = read_face_crop,
) -> tuple[TiltedHardPositiveFeature, ...]:
    overlay = GodModeOverlay(width=1280, height=720)
    owner_embeddings = np.load(owner_embedding_path)
    overlay.set_owner_embedding(owner_embeddings)
    detector = overlay._detector
    recognizer = overlay._recognizer
    if detector is None or recognizer is None:
        raise RuntimeError("overlay detector/recognizer unavailable")

    features: list[TiltedHardPositiveFeature] = []
    for path in iter_image_paths(source_dir):
        frame = read_image(path)
        if frame is None:
            continue
        embedding = extract_crop_embedding(overlay, frame)
        if embedding is None:
            continue
        owner_label, owner_score = classify_owner_embedding(
            recognizer=recognizer,
            owner_embeddings=owner_embeddings,
            embedding=embedding.reshape(1, -1),
            face_confidence=1.0,
        )
        del owner_label
        roll_degrees, detection_score = detect_primary_face_roll_degrees(
            frame=frame,
            detector=detector,
        )
        if roll_degrees is None:
            continue
        features.append(
            TiltedHardPositiveFeature(
                source_path=path,
                roll_degrees=roll_degrees,
                abs_roll_degrees=abs(roll_degrees),
                detection_score=detection_score,
                owner_score=float(owner_score),
                embedding=np.asarray(embedding, dtype=np.float32).reshape(-1),
            )
        )
    return tuple(features)


def detect_primary_face_roll_degrees(
    *,
    frame: FrameArray,
    detector: DetectorLike,
) -> tuple[float | None, float]:
    padded = _pad_frame_for_roll_detection(frame)
    set_detector_input_size(detector, (int(padded.shape[1]), int(padded.shape[0])))
    _, detections = detector.detect(padded)
    if detections is None or len(detections) == 0:
        return None, 0.0
    best = max(detections, key=lambda row: float(row[14]))
    roll_degrees = estimate_eye_line_roll_degrees(best)
    return roll_degrees, float(best[14])


def _pad_frame_for_roll_detection(frame: FrameArray, *, pad_ratio: float = 0.20) -> FrameArray:
    height, width = frame.shape[:2]
    pad_y = max(4, int(height * pad_ratio))
    pad_x = max(4, int(width * pad_ratio))
    return cast(
        FrameArray,
        cv2.copyMakeBorder(
            frame,
            pad_y,
            pad_y,
            pad_x,
            pad_x,
            cv2.BORDER_REFLECT_101,
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    source_dir = Path(args.source)
    output_root = Path(args.output_root)
    if output_root == DEFAULT_OUTPUT_ROOT:
        output_root = output_root / source_dir.parent.name

    features = build_tilted_hard_positive_features(
        source_dir=source_dir,
        owner_embedding_path=Path(args.owner_embedding_path),
    )
    selected = select_tilted_hard_positive_features(
        features,
        min_abs_roll_deg=float(args.min_abs_roll_deg),
        min_detection_score=float(args.min_detection_score),
        min_owner_score=float(args.min_owner_score),
        max_similarity_to_selected=float(args.max_similarity_to_selected),
        max_selected=None if args.max_selected is None else int(args.max_selected),
    )
    result = copy_tilted_hard_positive_features(
        source_root=source_dir,
        selected_features=selected,
        output_root=output_root,
    )
    print(f"source={source_dir}")
    print(f"output_root={result.output_root}")
    print(f"selected={result.total_selected}")
    print(f"manifest={result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

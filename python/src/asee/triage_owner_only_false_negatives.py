"""Conservative triage for owner-only false-negative candidate buckets."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2
from typing import Any, cast

import cv2
import numpy as np

from .enroll_owner import DEFAULT_OWNER_EMBED_PATH
from .overlay import GodModeOverlay
from .owner_policy import classify_owner_embedding
from .retrain_owner_embedding import (
    DEFAULT_FALSE_NEGATIVE_DIR,
    build_cosine_similarity_matrix,
    extract_crop_embedding,
    iter_image_paths,
    read_face_crop,
)

DEFAULT_SOURCE_DIR = Path("/home/yuiseki/Workspaces/private/datasets/faces/others")
DEFAULT_OUTPUT_ROOT = Path(
    "/home/yuiseki/Workspaces/private/datasets/faces/others_owner_only_triaged"
)
DEFAULT_MIN_FACE_SIZE = 72
DEFAULT_MIN_BLUR_VARIANCE = 60.0
DEFAULT_MIN_DETECTION_SCORE = 0.42
DEFAULT_MIN_OWNER_SCORE = 0.45
DEFAULT_MIN_OWNER_FALSE_NEGATIVE_SIMILARITY = 0.52

type EmbeddingArray = np.ndarray
type FrameArray = np.ndarray


@dataclass(frozen=True, slots=True)
class OwnerOnlyTriageThresholds:
    min_face_size: int = DEFAULT_MIN_FACE_SIZE
    min_blur_variance: float = DEFAULT_MIN_BLUR_VARIANCE
    min_detection_score: float = DEFAULT_MIN_DETECTION_SCORE
    min_owner_score: float = DEFAULT_MIN_OWNER_SCORE
    min_owner_false_negative_similarity: float = (
        DEFAULT_MIN_OWNER_FALSE_NEGATIVE_SIMILARITY
    )


@dataclass(frozen=True, slots=True)
class OwnerOnlyCandidateFeature:
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


@dataclass(frozen=True, slots=True)
class OwnerOnlyTriageResult:
    output_root: Path
    manifest_path: Path
    total_files: int
    counts: dict[str, int]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Triages owner-only SUBJECT captures into likely owner false negatives, "
            "low-quality crops, and uncertain examples."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=f"Source SUBJECT bucket (default: {DEFAULT_SOURCE_DIR})",
    )
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
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Destination root for triaged copies. Defaults to "
            "<DEFAULT_OUTPUT_ROOT>/<source leaf>."
        ),
    )
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=DEFAULT_MIN_FACE_SIZE,
    )
    parser.add_argument(
        "--min-blur-variance",
        type=float,
        default=DEFAULT_MIN_BLUR_VARIANCE,
    )
    parser.add_argument(
        "--min-detection-score",
        type=float,
        default=DEFAULT_MIN_DETECTION_SCORE,
    )
    parser.add_argument(
        "--min-owner-score",
        type=float,
        default=DEFAULT_MIN_OWNER_SCORE,
    )
    parser.add_argument(
        "--min-owner-false-negative-similarity",
        type=float,
        default=DEFAULT_MIN_OWNER_FALSE_NEGATIVE_SIMILARITY,
    )
    return parser


def classify_owner_only_candidate(
    feature: OwnerOnlyCandidateFeature,
    *,
    thresholds: OwnerOnlyTriageThresholds,
) -> str:
    if feature.metadata_label != "SUBJECT":
        return "uncertain"
    if not _is_single_subject_frame(feature):
        return "uncertain"
    if (
        feature.min_face_size < thresholds.min_face_size
        or feature.blur_variance < thresholds.min_blur_variance
        or feature.detection_score < thresholds.min_detection_score
    ):
        return "low_quality"
    if (
        feature.owner_score >= thresholds.min_owner_score
        and feature.owner_false_negative_similarity
        >= thresholds.min_owner_false_negative_similarity
    ):
        return "likely_owner_false_negative"
    return "uncertain"


def triage_owner_only_candidates(
    *,
    source_root: Path,
    features: Sequence[OwnerOnlyCandidateFeature],
    output_root: Path,
    thresholds: OwnerOnlyTriageThresholds,
    copy_file: Callable[[Path, Path], object] | None = None,
) -> OwnerOnlyTriageResult:
    output_root.mkdir(parents=True, exist_ok=True)
    writer = copy_file or copy2
    counts = {
        "likely_owner_false_negative": 0,
        "low_quality": 0,
        "uncertain": 0,
    }
    manifest_path = output_root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for feature in features:
            bucket = classify_owner_only_candidate(feature, thresholds=thresholds)
            counts[bucket] += 1
            relative_path = feature.source_path.relative_to(source_root)
            destination = output_root / bucket / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            writer(feature.source_path, destination)
            sidecar_path = feature.source_path.with_suffix(".json")
            if sidecar_path.exists():
                writer(sidecar_path, destination.with_suffix(".json"))
            manifest.write(
                json.dumps(
                    {
                        "source_path": str(feature.source_path),
                        "relative_path": str(relative_path),
                        "bucket": bucket,
                        "owner_score": feature.owner_score,
                        "owner_false_negative_similarity": feature.owner_false_negative_similarity,
                        "min_face_size": feature.min_face_size,
                        "blur_variance": feature.blur_variance,
                        "detection_score": feature.detection_score,
                        "metadata_label": feature.metadata_label,
                        "owner_count": feature.owner_count,
                        "subject_count": feature.subject_count,
                        "people_count": feature.people_count,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            manifest.write("\n")
    return OwnerOnlyTriageResult(
        output_root=output_root,
        manifest_path=manifest_path,
        total_files=len(features),
        counts=counts,
    )


def build_owner_only_candidate_features(
    *,
    source_dir: Path,
    owner_false_negative_dir: Path,
    owner_embedding_path: Path,
    read_image: Callable[[Path], FrameArray | None] = read_face_crop,
) -> tuple[OwnerOnlyCandidateFeature, ...]:
    overlay = GodModeOverlay(width=1280, height=720)
    owner_embeddings = np.load(owner_embedding_path)
    overlay.set_owner_embedding(owner_embeddings)

    def _collect_embeddings(root: Path) -> tuple[list[Path], EmbeddingArray]:
        paths: list[Path] = []
        embeddings: list[EmbeddingArray] = []
        for path in iter_image_paths(root):
            frame = read_image(path)
            if frame is None:
                continue
            embedding = extract_crop_embedding(overlay, frame)
            if embedding is None:
                continue
            paths.append(path)
            embeddings.append(cast(EmbeddingArray, embedding.reshape(-1)))
        if not embeddings:
            raise RuntimeError(f"no usable embeddings found under {root}")
        return paths, np.stack(embeddings, axis=0).astype(np.float32, copy=False)

    _, owner_false_negative_embeddings = _collect_embeddings(owner_false_negative_dir)
    source_paths, source_embeddings = _collect_embeddings(source_dir)
    source_to_owner_false_negative = build_cosine_similarity_matrix(
        owner_false_negative_embeddings,
        source_embeddings,
    ).max(axis=0)

    features: list[OwnerOnlyCandidateFeature] = []
    for path, embedding, owner_false_negative_similarity in zip(
        source_paths,
        source_embeddings,
        source_to_owner_false_negative,
        strict=False,
    ):
        frame = read_image(path)
        if frame is None:
            continue
        sidecar_payload = _load_sidecar_payload(path.with_suffix(".json"))
        label, owner_score = classify_owner_embedding(
            recognizer=overlay._recognizer,
            owner_embeddings=owner_embeddings,
            embedding=embedding.reshape(1, -1),
            face_confidence=float(sidecar_payload.get("score", 0.0)),
        )
        del label
        frame_counts = cast(dict[str, Any], sidecar_payload.get("frameCounts") or {})
        features.append(
            OwnerOnlyCandidateFeature(
                source_path=path,
                owner_score=float(owner_score),
                owner_false_negative_similarity=float(owner_false_negative_similarity),
                min_face_size=min(frame.shape[:2]),
                blur_variance=_blur_variance(frame),
                detection_score=_resolve_detection_score(path, sidecar_payload),
                metadata_label=str(sidecar_payload.get("label", "")),
                owner_count=_safe_int(frame_counts.get("ownerCount")),
                subject_count=_safe_int(frame_counts.get("subjectCount")),
                people_count=_safe_int(frame_counts.get("peopleCount")),
            )
        )
    return tuple(features)


def _is_single_subject_frame(feature: OwnerOnlyCandidateFeature) -> bool:
    return (
        feature.subject_count == 1
        and feature.people_count == 1
        and (feature.owner_count is None or feature.owner_count == 0)
    )


def _load_sidecar_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _resolve_detection_score(path: Path, sidecar_payload: dict[str, Any]) -> float:
    if "score" in sidecar_payload:
        try:
            return float(sidecar_payload["score"])
        except Exception:
            pass
    match = re.search(r"score([0-9]+\.[0-9]+)", path.name)
    if match:
        return float(match.group(1))
    return 0.0


def _blur_variance(frame: FrameArray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    thresholds = OwnerOnlyTriageThresholds(
        min_face_size=int(args.min_face_size),
        min_blur_variance=float(args.min_blur_variance),
        min_detection_score=float(args.min_detection_score),
        min_owner_score=float(args.min_owner_score),
        min_owner_false_negative_similarity=float(
            args.min_owner_false_negative_similarity
        ),
    )
    source = Path(args.source)
    output_root = (
        Path(args.output_root)
        if args.output_root is not None
        else DEFAULT_OUTPUT_ROOT / source.name
    )
    features = build_owner_only_candidate_features(
        source_dir=source,
        owner_false_negative_dir=Path(args.owner_false_negative_dir),
        owner_embedding_path=Path(args.owner_embedding_path),
    )
    result = triage_owner_only_candidates(
        source_root=source,
        features=features,
        output_root=output_root,
        thresholds=thresholds,
    )
    print(f"output_root={result.output_root}")
    print(f"manifest={result.manifest_path}")
    print(f"total_files={result.total_files}")
    for bucket, count in result.counts.items():
        print(f"{bucket}={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

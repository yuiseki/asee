"""Conservative auto-triage for mixed SUBJECT session buckets."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2
from typing import cast

import numpy as np

from .enroll_owner import DEFAULT_OWNER_EMBED_PATH
from .overlay import GodModeOverlay
from .owner_policy import classify_owner_embedding
from .retrain_owner_embedding import (
    DEFAULT_FALSE_NEGATIVE_DIR,
    DEFAULT_GUEST_SESSION_ROOT,
    build_cosine_similarity_matrix,
    extract_crop_embedding,
    iter_image_paths,
    read_face_crop,
)

DEFAULT_SESSION_DIR = DEFAULT_GUEST_SESSION_ROOT / "2026-03-13_22-41-56"
DEFAULT_TRIAGE_ROOT = Path(
    "/home/yuiseki/Workspaces/private/datasets/faces/others_guest_session_triaged"
)
DEFAULT_MAX_OWNER_SCORE_FOR_LIKELY_GUEST = 0.42
DEFAULT_MAX_OWNER_FALSE_NEGATIVE_SIMILARITY_FOR_LIKELY_GUEST = 0.55
DEFAULT_MIN_OWNER_SCORE_FOR_LIKELY_OWNER_FALSE_NEGATIVE = 0.52
DEFAULT_MIN_OWNER_FALSE_NEGATIVE_SIMILARITY_FOR_LIKELY_OWNER_FALSE_NEGATIVE = 0.80

type EmbeddingArray = np.ndarray
type FrameArray = np.ndarray

BucketName = str


@dataclass(frozen=True, slots=True)
class SessionTriageThresholds:
    max_owner_score_for_likely_guest: float = DEFAULT_MAX_OWNER_SCORE_FOR_LIKELY_GUEST
    max_owner_false_negative_similarity_for_likely_guest: float = (
        DEFAULT_MAX_OWNER_FALSE_NEGATIVE_SIMILARITY_FOR_LIKELY_GUEST
    )
    min_owner_score_for_likely_owner_false_negative: float = (
        DEFAULT_MIN_OWNER_SCORE_FOR_LIKELY_OWNER_FALSE_NEGATIVE
    )
    min_owner_false_negative_similarity_for_likely_owner_false_negative: float = (
        DEFAULT_MIN_OWNER_FALSE_NEGATIVE_SIMILARITY_FOR_LIKELY_OWNER_FALSE_NEGATIVE
    )


@dataclass(frozen=True, slots=True)
class SubjectSessionFeature:
    source_path: Path
    owner_score: float
    owner_false_negative_similarity: float


@dataclass(frozen=True, slots=True)
class SessionTriageResult:
    output_root: Path
    manifest_path: Path
    total_files: int
    counts: dict[str, int]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for mixed subject session triage."""
    parser = argparse.ArgumentParser(
        description=(
            "Split one mixed SUBJECT session bucket into likely guest negatives, "
            "likely owner false negatives, and uncertain examples."
        )
    )
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=DEFAULT_SESSION_DIR,
        help=f"Mixed session directory to triage (default: {DEFAULT_SESSION_DIR})",
    )
    parser.add_argument(
        "--owner-false-negative-dir",
        type=Path,
        default=DEFAULT_FALSE_NEGATIVE_DIR,
        help=(
            "Reference OWNER false-negative directory used as an owner-like bank "
            f"(default: {DEFAULT_FALSE_NEGATIVE_DIR})"
        ),
    )
    parser.add_argument(
        "--owner-embedding-path",
        type=Path,
        default=DEFAULT_OWNER_EMBED_PATH,
        help=f"Current live owner embedding bank (default: {DEFAULT_OWNER_EMBED_PATH})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Destination root for triaged copies. Defaults to "
            "<DEFAULT_TRIAGE_ROOT>/<session name>."
        ),
    )
    parser.add_argument(
        "--max-owner-score-for-likely-guest",
        type=float,
        default=DEFAULT_MAX_OWNER_SCORE_FOR_LIKELY_GUEST,
    )
    parser.add_argument(
        "--max-owner-fn-similarity-for-likely-guest",
        type=float,
        default=DEFAULT_MAX_OWNER_FALSE_NEGATIVE_SIMILARITY_FOR_LIKELY_GUEST,
    )
    parser.add_argument(
        "--min-owner-score-for-likely-owner-false-negative",
        type=float,
        default=DEFAULT_MIN_OWNER_SCORE_FOR_LIKELY_OWNER_FALSE_NEGATIVE,
    )
    parser.add_argument(
        "--min-owner-fn-similarity-for-likely-owner-false-negative",
        type=float,
        default=DEFAULT_MIN_OWNER_FALSE_NEGATIVE_SIMILARITY_FOR_LIKELY_OWNER_FALSE_NEGATIVE,
    )
    return parser


def classify_subject_session_feature(
    feature: SubjectSessionFeature,
    *,
    thresholds: SessionTriageThresholds,
) -> BucketName:
    """Assign one feature into a conservative triage bucket."""
    if (
        feature.owner_score <= thresholds.max_owner_score_for_likely_guest
        and feature.owner_false_negative_similarity
        <= thresholds.max_owner_false_negative_similarity_for_likely_guest
    ):
        return "likely_guest_negative"
    if (
        feature.owner_score >= thresholds.min_owner_score_for_likely_owner_false_negative
        and feature.owner_false_negative_similarity
        >= thresholds.min_owner_false_negative_similarity_for_likely_owner_false_negative
    ):
        return "likely_owner_false_negative"
    return "uncertain"


def triage_subject_session_features(
    *,
    session_root: Path,
    features: Sequence[SubjectSessionFeature],
    output_root: Path,
    thresholds: SessionTriageThresholds,
    copy_file: Callable[[Path, Path], object] | None = None,
) -> SessionTriageResult:
    """Copy one mixed session into triaged bucket directories with a JSONL manifest."""
    output_root.mkdir(parents=True, exist_ok=True)
    writer = copy_file or copy2
    counts = {
        "likely_guest_negative": 0,
        "likely_owner_false_negative": 0,
        "uncertain": 0,
    }
    manifest_path = output_root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for feature in features:
            bucket = classify_subject_session_feature(feature, thresholds=thresholds)
            counts[bucket] += 1
            relative_path = feature.source_path.relative_to(session_root)
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
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            manifest.write("\n")
    return SessionTriageResult(
        output_root=output_root,
        manifest_path=manifest_path,
        total_files=len(features),
        counts=counts,
    )


def build_subject_session_features(
    *,
    session_dir: Path,
    owner_false_negative_dir: Path,
    owner_embedding_path: Path,
    read_image: Callable[[Path], FrameArray | None] = read_face_crop,
) -> tuple[SubjectSessionFeature, ...]:
    """Extract owner similarity features for one mixed session directory."""
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

    owner_false_negative_paths, owner_false_negative_embeddings = _collect_embeddings(
        owner_false_negative_dir
    )
    del owner_false_negative_paths

    session_paths, session_embeddings = _collect_embeddings(session_dir)
    session_to_owner_false_negative = build_cosine_similarity_matrix(
        owner_false_negative_embeddings,
        session_embeddings,
    ).max(axis=0)

    features: list[SubjectSessionFeature] = []
    for path, embedding, owner_false_negative_similarity in zip(
        session_paths,
        session_embeddings,
        session_to_owner_false_negative,
        strict=False,
    ):
        label, owner_score = classify_owner_embedding(
            recognizer=overlay._recognizer,
            owner_embeddings=owner_embeddings,
            embedding=embedding.reshape(1, -1),
            face_confidence=1.0,
        )
        del label
        features.append(
            SubjectSessionFeature(
                source_path=path,
                owner_score=float(owner_score),
                owner_false_negative_similarity=float(owner_false_negative_similarity),
            )
        )
    return tuple(features)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for mixed subject session triage."""
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    thresholds = SessionTriageThresholds(
        max_owner_score_for_likely_guest=float(args.max_owner_score_for_likely_guest),
        max_owner_false_negative_similarity_for_likely_guest=float(
            args.max_owner_fn_similarity_for_likely_guest
        ),
        min_owner_score_for_likely_owner_false_negative=float(
            args.min_owner_score_for_likely_owner_false_negative
        ),
        min_owner_false_negative_similarity_for_likely_owner_false_negative=float(
            args.min_owner_fn_similarity_for_likely_owner_false_negative
        ),
    )
    session_dir = Path(args.session_dir)
    output_root = (
        Path(args.output_root)
        if args.output_root is not None
        else DEFAULT_TRIAGE_ROOT / session_dir.name
    )
    features = build_subject_session_features(
        session_dir=session_dir,
        owner_false_negative_dir=Path(args.owner_false_negative_dir),
        owner_embedding_path=Path(args.owner_embedding_path),
    )
    result = triage_subject_session_features(
        session_root=session_dir,
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

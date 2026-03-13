"""Offline retraining helpers for owner embeddings."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from shutil import copy2
from typing import Protocol, cast

import cv2
import numpy as np
import numpy.typing as npt

from .enroll_owner import DEFAULT_OWNER_EMBED_PATH
from .overlay import GodModeOverlay
from .owner_policy import FaceRecognizerLike, classify_owner_embedding
from .tracking import FaceBox

DEFAULT_FALSE_NEGATIVE_DIR = Path(
    "/home/yuiseki/Workspaces/private/datasets/faces/owner_false_negative"
)
DEFAULT_GUEST_SESSION_ROOT = Path(
    "/home/yuiseki/Workspaces/private/datasets/faces/others_guest_session"
)
DEFAULT_SNAPSHOT_DIR = Path(
    "/home/yuiseki/Workspaces/private/datasets/faces/owner_embedding_snapshots"
)

type EmbeddingArray = npt.NDArray[np.float32]
type FrameArray = npt.NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class DatasetEvaluation:
    total_files: int
    usable_embeddings: int
    skipped_files: int
    owner_predictions: int
    subject_predictions: int
    mean_score: float


@dataclass(frozen=True, slots=True)
class RetrainValidationReport:
    owner_embedding_path: Path
    snapshot_path: Path
    false_negative_dir: Path
    negative_validation_dir: Path
    before_positive: DatasetEvaluation
    after_positive: DatasetEvaluation
    before_negative: DatasetEvaluation
    after_negative: DatasetEvaluation
    added_embeddings: int


class OverlayLike(Protocol):
    _recognizer: FaceRecognizerLike | None

    def extract_embedding(self, frame: FrameArray, face_box: FaceBox) -> EmbeddingArray | None: ...


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for owner embedding retraining."""
    parser = argparse.ArgumentParser(description="Retrain owner embeddings from false negatives")
    parser.add_argument(
        "--owner-embedding-path",
        type=Path,
        default=DEFAULT_OWNER_EMBED_PATH,
    )
    parser.add_argument(
        "--false-negative-dir",
        type=Path,
        default=DEFAULT_FALSE_NEGATIVE_DIR,
    )
    parser.add_argument(
        "--negative-validation-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--guest-session-root",
        type=Path,
        default=DEFAULT_GUEST_SESSION_ROOT,
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=DEFAULT_SNAPSHOT_DIR,
    )
    return parser


def normalize_owner_embeddings(embeddings: EmbeddingArray) -> EmbeddingArray:
    """Normalize embeddings into the runtime `(N, 1, 128)` shape."""
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim == 1:
        return cast(EmbeddingArray, array.reshape(1, 1, -1))
    if array.ndim == 2:
        return cast(EmbeddingArray, array.reshape(array.shape[0], 1, array.shape[1]))
    if array.ndim == 3:
        return array.copy()
    raise ValueError(f"unsupported embedding shape: {array.shape}")


def augment_owner_embeddings(
    *,
    current: EmbeddingArray,
    additions: EmbeddingArray,
) -> EmbeddingArray:
    """Append normalized additions to the current owner embedding bank."""
    normalized_current = normalize_owner_embeddings(current)
    normalized_additions = normalize_owner_embeddings(additions)
    return np.concatenate([normalized_current, normalized_additions], axis=0)


def snapshot_owner_embedding(
    *,
    owner_embedding_path: Path,
    snapshot_dir: Path,
    timestamp: str,
) -> Path:
    """Copy the current owner embedding into a timestamped snapshot."""
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / f"owner_embedding_{timestamp}.npy"
    copy2(owner_embedding_path, snapshot_path)
    return snapshot_path


def resolve_latest_guest_session_dir(root: Path) -> Path:
    """Return the latest guest session directory under the archive root."""
    candidates = sorted(path for path in root.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"no guest session directories under {root}")
    return candidates[-1]


def iter_image_paths(root: Path) -> list[Path]:
    """Return stable, recursive image paths under a dataset root."""
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file())


def read_face_crop(path: Path) -> FrameArray | None:
    """Load one cropped face image from disk."""
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return cast(FrameArray | None, frame)


def build_crop_face_box(frame: FrameArray) -> FaceBox:
    """Treat the whole cropped face image as one face box."""
    height, width = frame.shape[:2]
    return FaceBox(x=0, y=0, w=width, h=height, confidence=1.0)


def extract_crop_embedding(overlay: OverlayLike, frame: FrameArray) -> EmbeddingArray | None:
    """Extract one embedding from a pre-cropped face image."""
    return overlay.extract_embedding(frame, build_crop_face_box(frame))


def evaluate_image_paths(
    *,
    image_paths: Sequence[Path],
    read_image: Callable[[Path], FrameArray | None],
    extract_embedding: Callable[[FrameArray], EmbeddingArray | None],
    classify_embedding: Callable[[Path, EmbeddingArray], tuple[str, float]],
) -> DatasetEvaluation:
    """Evaluate how one owner embedding bank classifies a dataset."""
    total_files = len(image_paths)
    usable_embeddings = 0
    skipped_files = 0
    owner_predictions = 0
    subject_predictions = 0
    scores: list[float] = []

    for path in image_paths:
        frame = read_image(path)
        if frame is None:
            skipped_files += 1
            continue
        embedding = extract_embedding(frame)
        if embedding is None:
            skipped_files += 1
            continue
        usable_embeddings += 1
        label, score = classify_embedding(path, embedding)
        scores.append(score)
        if label == "OWNER":
            owner_predictions += 1
        else:
            subject_predictions += 1

    mean_score = float(np.mean(scores)) if scores else 0.0
    return DatasetEvaluation(
        total_files=total_files,
        usable_embeddings=usable_embeddings,
        skipped_files=skipped_files,
        owner_predictions=owner_predictions,
        subject_predictions=subject_predictions,
        mean_score=mean_score,
    )


def collect_dataset_embeddings(
    *,
    image_paths: Sequence[Path],
    read_image: Callable[[Path], FrameArray | None],
    extract_embedding: Callable[[FrameArray], EmbeddingArray | None],
) -> EmbeddingArray:
    """Collect usable embeddings from one dataset."""
    embeddings: list[EmbeddingArray] = []
    for path in image_paths:
        frame = read_image(path)
        if frame is None:
            continue
        embedding = extract_embedding(frame)
        if embedding is None:
            continue
        embeddings.append(normalize_owner_embeddings(embedding)[0])
    if not embeddings:
        raise RuntimeError("no usable embeddings extracted from false-negative dataset")
    return cast(EmbeddingArray, np.stack(embeddings, axis=0))


def run_retraining(
    *,
    owner_embedding_path: Path,
    false_negative_dir: Path,
    negative_validation_dir: Path,
    snapshot_dir: Path,
    overlay: OverlayLike | None = None,
    read_image: Callable[[Path], FrameArray | None] = read_face_crop,
) -> RetrainValidationReport:
    """Snapshot, retrain, validate, and write the updated owner embeddings."""
    active_overlay: OverlayLike
    if overlay is not None:
        active_overlay = overlay
    else:
        active_overlay = cast(OverlayLike, GodModeOverlay(width=1280, height=720))
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    snapshot_path = snapshot_owner_embedding(
        owner_embedding_path=owner_embedding_path,
        snapshot_dir=snapshot_dir,
        timestamp=timestamp,
    )

    current_embeddings = normalize_owner_embeddings(
        cast(EmbeddingArray, np.load(owner_embedding_path))
    )
    positive_paths = iter_image_paths(false_negative_dir)
    negative_paths = iter_image_paths(negative_validation_dir)
    if not positive_paths:
        raise RuntimeError(f"no images found under {false_negative_dir}")
    if not negative_paths:
        raise RuntimeError(f"no images found under {negative_validation_dir}")

    def extract_embedding(frame: FrameArray) -> EmbeddingArray | None:
        return extract_crop_embedding(active_overlay, frame)

    def classify_with(
        owner_embeddings: EmbeddingArray,
    ) -> Callable[[Path, EmbeddingArray], tuple[str, float]]:
        def _classify(path: Path, embedding: EmbeddingArray) -> tuple[str, float]:
            del path
            return classify_owner_embedding(
                recognizer=active_overlay._recognizer,
                owner_embeddings=owner_embeddings,
                embedding=embedding,
                face_confidence=1.0,
            )

        return _classify

    before_positive = evaluate_image_paths(
        image_paths=positive_paths,
        read_image=read_image,
        extract_embedding=extract_embedding,
        classify_embedding=classify_with(current_embeddings),
    )
    before_negative = evaluate_image_paths(
        image_paths=negative_paths,
        read_image=read_image,
        extract_embedding=extract_embedding,
        classify_embedding=classify_with(current_embeddings),
    )

    additions = collect_dataset_embeddings(
        image_paths=positive_paths,
        read_image=read_image,
        extract_embedding=extract_embedding,
    )
    candidate_embeddings = augment_owner_embeddings(current=current_embeddings, additions=additions)

    after_positive = evaluate_image_paths(
        image_paths=positive_paths,
        read_image=read_image,
        extract_embedding=extract_embedding,
        classify_embedding=classify_with(candidate_embeddings),
    )
    after_negative = evaluate_image_paths(
        image_paths=negative_paths,
        read_image=read_image,
        extract_embedding=extract_embedding,
        classify_embedding=classify_with(candidate_embeddings),
    )

    np.save(owner_embedding_path, candidate_embeddings)
    return RetrainValidationReport(
        owner_embedding_path=owner_embedding_path,
        snapshot_path=snapshot_path,
        false_negative_dir=false_negative_dir,
        negative_validation_dir=negative_validation_dir,
        before_positive=before_positive,
        after_positive=after_positive,
        before_negative=before_negative,
        after_negative=after_negative,
        added_embeddings=int(candidate_embeddings.shape[0] - current_embeddings.shape[0]),
    )


def format_evaluation(name: str, evaluation: DatasetEvaluation) -> str:
    """Format one dataset evaluation for CLI output."""
    return (
        f"{name}: total={evaluation.total_files} usable={evaluation.usable_embeddings} "
        f"skipped={evaluation.skipped_files} owner={evaluation.owner_predictions} "
        f"subject={evaluation.subject_predictions} mean_score={evaluation.mean_score:.4f}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for owner embedding retraining."""
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    negative_validation_dir = (
        Path(args.negative_validation_dir)
        if args.negative_validation_dir is not None
        else resolve_latest_guest_session_dir(Path(args.guest_session_root))
    )
    report = run_retraining(
        owner_embedding_path=Path(args.owner_embedding_path),
        false_negative_dir=Path(args.false_negative_dir),
        negative_validation_dir=negative_validation_dir,
        snapshot_dir=Path(args.snapshot_dir),
    )
    print(f"snapshot={report.snapshot_path}")
    print(f"owner_embedding={report.owner_embedding_path}")
    print(f"false_negative_dir={report.false_negative_dir}")
    print(f"negative_validation_dir={report.negative_validation_dir}")
    print(f"added_embeddings={report.added_embeddings}")
    print(format_evaluation("before_positive", report.before_positive))
    print(format_evaluation("after_positive", report.after_positive))
    print(format_evaluation("before_negative", report.before_negative))
    print(format_evaluation("after_negative", report.after_negative))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

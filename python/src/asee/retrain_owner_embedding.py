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
from .owner_policy import (
    OWNER_COSINE_THRESHOLD,
    OWNER_TOPK,
    FaceRecognizerLike,
    classify_owner_embedding,
)
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
    selected_false_negative_paths: tuple[Path, ...] = ()
    candidate_embedding_path: Path | None = None


@dataclass(frozen=True, slots=True)
class GreedySelectionResult:
    selected_indices: tuple[int, ...]
    selected_paths: tuple[Path, ...]


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
    parser.add_argument(
        "--selection-mode",
        choices=("negative-aware-greedy", "full-add"),
        default="negative-aware-greedy",
    )
    parser.add_argument(
        "--negative-penalty",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--max-selected",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the candidate owner embedding back into the live model path.",
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
    image_suffixes = {".jpg", ".jpeg", ".png"}
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in image_suffixes
    )


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


def build_cosine_similarity_matrix(
    references: EmbeddingArray,
    samples: EmbeddingArray,
) -> npt.NDArray[np.float32]:
    """Compute cosine similarity matrix between reference and sample embeddings."""
    flat_references = normalize_owner_embeddings(references).reshape(len(references), -1)
    flat_samples = normalize_owner_embeddings(samples).reshape(len(samples), -1)
    ref_norm = np.linalg.norm(flat_references, axis=1, keepdims=True)
    sample_norm = np.linalg.norm(flat_samples, axis=1, keepdims=True)
    safe_ref = np.where(ref_norm == 0, 1.0, ref_norm)
    safe_sample = np.where(sample_norm == 0, 1.0, sample_norm)
    normalized_refs = flat_references / safe_ref
    normalized_samples = flat_samples / safe_sample
    return cast(
        npt.NDArray[np.float32],
        (normalized_refs @ normalized_samples.T).astype(np.float32, copy=False),
    )


def build_topk_values(
    similarities: npt.NDArray[np.float32],
    *,
    topk: int,
) -> npt.NDArray[np.float32]:
    """Build per-sample ascending top-k similarity buffers."""
    if similarities.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    k = min(topk, similarities.shape[0])
    partitioned = np.partition(similarities, similarities.shape[0] - k, axis=0)[-k:]
    return np.sort(partitioned, axis=0).T.copy()


def greedy_select_false_negative_candidates(
    *,
    candidate_paths: Sequence[Path],
    positive_candidate_scores: npt.NDArray[np.float32],
    negative_candidate_scores: npt.NDArray[np.float32],
    positive_topk_values: npt.NDArray[np.float32],
    negative_topk_values: npt.NDArray[np.float32],
    threshold: float,
    negative_penalty: float,
    max_selected: int | None = None,
) -> GreedySelectionResult:
    """Greedily add candidates that improve positives more than they harm negatives."""
    if len(candidate_paths) != positive_candidate_scores.shape[0]:
        raise ValueError("candidate_paths and positive_candidate_scores must align")
    if len(candidate_paths) != negative_candidate_scores.shape[0]:
        raise ValueError("candidate_paths and negative_candidate_scores must align")

    selected_indices: list[int] = []
    remaining = set(range(len(candidate_paths)))
    current_positive = positive_topk_values.copy()
    current_negative = negative_topk_values.copy()

    while remaining:
        if max_selected is not None and len(selected_indices) >= max_selected:
            break
        best_index: int | None = None
        best_utility = 0.0

        current_positive_owner = int(np.count_nonzero(current_positive.mean(axis=1) >= threshold))
        current_negative_owner = int(np.count_nonzero(current_negative.mean(axis=1) >= threshold))

        for index in sorted(remaining):
            trial_positive = apply_candidate_scores(
                topk_values=current_positive,
                candidate_scores=positive_candidate_scores[index],
            )
            trial_negative = apply_candidate_scores(
                topk_values=current_negative,
                candidate_scores=negative_candidate_scores[index],
            )
            positive_gain = (
                int(np.count_nonzero(trial_positive.mean(axis=1) >= threshold))
                - current_positive_owner
            )
            negative_gain = (
                int(np.count_nonzero(trial_negative.mean(axis=1) >= threshold))
                - current_negative_owner
            )
            utility = float(positive_gain) - negative_penalty * float(negative_gain)
            if utility > best_utility:
                best_utility = utility
                best_index = index

        if best_index is None:
            break

        selected_indices.append(best_index)
        remaining.remove(best_index)
        current_positive = apply_candidate_scores(
            topk_values=current_positive,
            candidate_scores=positive_candidate_scores[best_index],
        )
        current_negative = apply_candidate_scores(
            topk_values=current_negative,
            candidate_scores=negative_candidate_scores[best_index],
        )

    selected_paths = tuple(candidate_paths[index] for index in selected_indices)
    return GreedySelectionResult(
        selected_indices=tuple(selected_indices),
        selected_paths=selected_paths,
    )


def apply_candidate_scores(
    *,
    topk_values: npt.NDArray[np.float32],
    candidate_scores: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Insert one candidate similarity vector into per-sample top-k buffers."""
    if topk_values.size == 0:
        return topk_values.copy()
    updated = topk_values.copy()
    thresholds = updated[:, 0]
    improve_mask = candidate_scores > thresholds
    if not np.any(improve_mask):
        return updated
    updated[improve_mask, 0] = candidate_scores[improve_mask]
    updated[improve_mask] = np.sort(updated[improve_mask], axis=1)
    return updated


def run_retraining(
    *,
    owner_embedding_path: Path,
    false_negative_dir: Path,
    negative_validation_dir: Path,
    snapshot_dir: Path,
    selection_mode: str = "negative-aware-greedy",
    negative_penalty: float = 3.0,
    max_selected: int | None = None,
    apply: bool = False,
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

    positive_additions = collect_dataset_embeddings(
        image_paths=positive_paths,
        read_image=read_image,
        extract_embedding=extract_embedding,
    )
    selected_positive_paths = tuple(positive_paths)
    selected_additions = positive_additions
    if selection_mode == "negative-aware-greedy":
        positive_similarity = build_cosine_similarity_matrix(positive_additions, positive_additions)
        negative_embeddings = collect_dataset_embeddings(
            image_paths=negative_paths,
            read_image=read_image,
            extract_embedding=extract_embedding,
        )
        negative_similarity = build_cosine_similarity_matrix(
            positive_additions,
            negative_embeddings,
        )
        current_positive_similarity = build_cosine_similarity_matrix(
            current_embeddings,
            positive_additions,
        )
        current_negative_similarity = build_cosine_similarity_matrix(
            current_embeddings,
            negative_embeddings,
        )
        selection = greedy_select_false_negative_candidates(
            candidate_paths=positive_paths,
            positive_candidate_scores=positive_similarity,
            negative_candidate_scores=negative_similarity,
            positive_topk_values=build_topk_values(current_positive_similarity, topk=OWNER_TOPK),
            negative_topk_values=build_topk_values(current_negative_similarity, topk=OWNER_TOPK),
            threshold=OWNER_COSINE_THRESHOLD,
            negative_penalty=negative_penalty,
            max_selected=max_selected,
        )
        selected_positive_paths = selection.selected_paths
        if selection.selected_indices:
            selected_additions = positive_additions[list(selection.selected_indices)]
        else:
            selected_additions = cast(
                EmbeddingArray,
                np.zeros((0, 1, current_embeddings.shape[-1]), dtype=np.float32),
            )

    candidate_embeddings = (
        augment_owner_embeddings(current=current_embeddings, additions=selected_additions)
        if len(selected_additions) > 0
        else current_embeddings.copy()
    )

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

    candidate_snapshot_path = snapshot_dir / f"owner_embedding_candidate_{timestamp}.npy"
    candidate_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(candidate_snapshot_path, candidate_embeddings)
    if apply:
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
        selected_false_negative_paths=selected_positive_paths,
        candidate_embedding_path=candidate_snapshot_path,
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
        selection_mode=str(args.selection_mode),
        negative_penalty=float(args.negative_penalty),
        max_selected=None if args.max_selected is None else int(args.max_selected),
        apply=bool(args.apply),
    )
    print(f"snapshot={report.snapshot_path}")
    print(f"owner_embedding={report.owner_embedding_path}")
    print(f"candidate_embedding={report.candidate_embedding_path}")
    print(f"false_negative_dir={report.false_negative_dir}")
    print(f"negative_validation_dir={report.negative_validation_dir}")
    print(f"added_embeddings={report.added_embeddings}")
    print(f"selected_false_negatives={len(report.selected_false_negative_paths)}")
    print(format_evaluation("before_positive", report.before_positive))
    print(format_evaluation("after_positive", report.after_positive))
    print(format_evaluation("before_negative", report.before_negative))
    print(format_evaluation("after_negative", report.after_negative))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

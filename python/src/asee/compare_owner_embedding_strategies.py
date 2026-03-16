"""Compare append-vs-rebuild owner embedding strategies from labeled projects."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np

from .enroll_owner import DEFAULT_OWNER_EMBED_PATH
from .overlay import GodModeOverlay
from .owner_policy import (
    OWNER_COSINE_THRESHOLD,
    OWNER_TOPK,
    classify_owner_embedding,
)
from .retrain_owner_embedding import (
    DEFAULT_SNAPSHOT_DIR,
    DatasetEvaluation,
    EmbeddingArray,
    OverlayLike,
    build_cosine_similarity_matrix,
    build_topk_values,
    extract_crop_embedding,
    greedy_select_false_negative_candidates,
    normalize_owner_embeddings,
    read_face_crop,
    snapshot_owner_embedding,
)

DEFAULT_HARD_POSITIVE_BACKUP_ROOT = Path(
    "/home/yuiseki/Workspaces/private/datasets/faces/golden_review_backups/owner-golden-2026-03-15-v1"
)
DEFAULT_BASELINE_CONTACTS_BACKUP_ROOT = Path(
    "/home/yuiseki/Workspaces/private/datasets/faces/golden_review_backups/owner-baseline-contacts-v1"
)
DEFAULT_BASELINE_MAKEUP_BACKUP_ROOT = Path(
    "/home/yuiseki/Workspaces/private/datasets/faces/golden_review_backups/owner-baseline-makeup-v1"
)
DEFAULT_NON_FACE_HARD_NEGATIVE_BACKUP_ROOT = Path(
    "/home/yuiseki/Workspaces/private/datasets/faces/golden_review_backups/owner-non-face-hard-negatives-v1"
)
DEFAULT_BASELINE_HOLDOUT_BACKUP_ROOT = Path(
    "/home/yuiseki/Workspaces/private/datasets/faces/golden_review_backups/owner-baseline-holdout-v1"
)


@dataclass(frozen=True, slots=True)
class ReviewedSample:
    project_name: str
    label: str
    source_image: Path
    source_sidecar: Path | None
    task_id: int


@dataclass(frozen=True, slots=True)
class ReviewBundle:
    hard_positive_glasses: tuple[ReviewedSample, ...]
    baseline_contacts: tuple[ReviewedSample, ...]
    baseline_makeup: tuple[ReviewedSample, ...]
    non_face_owner_positives: tuple[ReviewedSample, ...]
    baseline_holdout: tuple[ReviewedSample, ...]
    guest_negative: tuple[ReviewedSample, ...]
    non_face_negative: tuple[ReviewedSample, ...]

    @property
    def append_candidates(self) -> tuple[ReviewedSample, ...]:
        return self.hard_positive_glasses

    @property
    def rebuild_sources(self) -> tuple[ReviewedSample, ...]:
        return (
            *self.hard_positive_glasses,
            *self.baseline_contacts,
            *self.baseline_makeup,
        )

    @property
    def negative_all(self) -> tuple[ReviewedSample, ...]:
        return (*self.guest_negative, *self.non_face_negative)


@dataclass(frozen=True, slots=True)
class StrategyEvaluationReport:
    bank_size: int
    source_images: int
    added_embeddings: int
    hard_positive_glasses: DatasetEvaluation
    baseline_contacts: DatasetEvaluation
    baseline_makeup: DatasetEvaluation
    non_face_owner_positives: DatasetEvaluation
    baseline_holdout: DatasetEvaluation
    guest_negative: DatasetEvaluation
    non_face_negative: DatasetEvaluation
    negative_all: DatasetEvaluation
    candidate_embedding_path: Path | None = None
    selected_source_paths: tuple[Path, ...] = ()


@dataclass(frozen=True, slots=True)
class StrategyComparisonReport:
    owner_embedding_path: Path
    snapshot_path: Path
    hard_positive_export: Path
    baseline_contacts_export: Path
    baseline_makeup_export: Path
    non_face_hard_negative_export: Path | None
    baseline_holdout_export: Path | None
    current: StrategyEvaluationReport
    append: StrategyEvaluationReport
    rebuild: StrategyEvaluationReport


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare append-vs-rebuild owner embedding strategies from labeled projects"
    )
    parser.add_argument(
        "--owner-embedding-path",
        type=Path,
        default=DEFAULT_OWNER_EMBED_PATH,
    )
    parser.add_argument(
        "--hard-positive-export",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--baseline-contacts-export",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--baseline-makeup-export",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--non-face-hard-negative-export",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--baseline-holdout-export",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=DEFAULT_SNAPSHOT_DIR,
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
    return parser


def resolve_latest_export_json(backup_root: Path) -> Path:
    candidates = sorted(path for path in backup_root.glob("*/export-json.json") if path.is_file())
    if not candidates:
        raise FileNotFoundError(f"no export-json.json found under {backup_root}")
    return candidates[-1]


def load_review_samples(export_json_path: Path, *, project_name: str) -> tuple[ReviewedSample, ...]:
    raw = json.loads(export_json_path.read_text(encoding="utf-8"))
    samples: list[ReviewedSample] = []
    for item in raw:
        annotations = item.get("annotations") or []
        if not annotations:
            continue
        annotation = annotations[-1]
        label = extract_choice_label(annotation.get("result") or [])
        if label is None:
            continue
        meta = item.get("meta") or {}
        source_image = meta.get("source_image")
        if not isinstance(source_image, str) or source_image == "":
            continue
        source_sidecar_value = meta.get("source_sidecar")
        source_sidecar = (
            Path(source_sidecar_value)
            if isinstance(source_sidecar_value, str) and source_sidecar_value != ""
            else None
        )
        samples.append(
            ReviewedSample(
                project_name=project_name,
                label=label,
                source_image=Path(source_image),
                source_sidecar=source_sidecar,
                task_id=int(item["id"]),
            )
        )
    return tuple(samples)


def extract_choice_label(results: list[object]) -> str | None:
    labels: list[str] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        value = result.get("value")
        if not isinstance(value, dict):
            continue
        choices = value.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if isinstance(choice, str) and choice:
                labels.append(choice)
    if not labels:
        return None
    unique_labels = list(dict.fromkeys(labels))
    if len(unique_labels) != 1:
        raise ValueError(f"expected one label choice, got {unique_labels}")
    return unique_labels[0]


def unique_samples_by_path(samples: tuple[ReviewedSample, ...]) -> tuple[ReviewedSample, ...]:
    seen: set[Path] = set()
    ordered: list[ReviewedSample] = []
    for sample in samples:
        if sample.source_image in seen:
            continue
        seen.add(sample.source_image)
        ordered.append(sample)
    return tuple(ordered)


def build_review_bundle(
    *,
    hard_positive_export: Path,
    baseline_contacts_export: Path,
    baseline_makeup_export: Path,
    non_face_hard_negative_export: Path | None = None,
    baseline_holdout_export: Path | None = None,
) -> ReviewBundle:
    hard_samples = load_review_samples(
        hard_positive_export,
        project_name="owner_hard_positive_glasses",
    )
    contacts_samples = load_review_samples(
        baseline_contacts_export,
        project_name="owner_baseline_contacts",
    )
    makeup_samples = load_review_samples(
        baseline_makeup_export,
        project_name="owner_baseline_makeup",
    )
    non_face_hard_negative_samples = (
        load_review_samples(
            non_face_hard_negative_export,
            project_name="owner_non_face_hard_negatives",
        )
        if non_face_hard_negative_export is not None
        else ()
    )
    baseline_holdout_samples = (
        load_review_samples(
            baseline_holdout_export,
            project_name="owner_baseline_holdout",
        )
        if baseline_holdout_export is not None
        else ()
    )
    all_samples = (
        *hard_samples,
        *contacts_samples,
        *makeup_samples,
        *non_face_hard_negative_samples,
        *baseline_holdout_samples,
    )
    return ReviewBundle(
        hard_positive_glasses=unique_samples_by_path(
            tuple(sample for sample in hard_samples if sample.label == "owner_positive")
        ),
        baseline_contacts=unique_samples_by_path(
            tuple(sample for sample in contacts_samples if sample.label == "owner_positive")
        ),
        baseline_makeup=unique_samples_by_path(
            tuple(sample for sample in makeup_samples if sample.label == "owner_positive")
        ),
        non_face_owner_positives=unique_samples_by_path(
            tuple(
                sample
                for sample in non_face_hard_negative_samples
                if sample.label == "owner_positive"
            )
        ),
        baseline_holdout=unique_samples_by_path(
            tuple(
                sample
                for sample in baseline_holdout_samples
                if sample.label == "owner_positive"
            )
        ),
        guest_negative=unique_samples_by_path(
            tuple(sample for sample in all_samples if sample.label == "guest_negative")
        ),
        non_face_negative=unique_samples_by_path(
            tuple(sample for sample in all_samples if sample.label == "non_face_negative")
        ),
    )


def compare_owner_embedding_strategies(
    *,
    owner_embedding_path: Path,
    hard_positive_export: Path,
    baseline_contacts_export: Path,
    baseline_makeup_export: Path,
    non_face_hard_negative_export: Path | None = None,
    baseline_holdout_export: Path | None = None,
    snapshot_dir: Path,
    negative_penalty: float = 3.0,
    max_selected: int | None = None,
    overlay: OverlayLike | None = None,
    embedding_lookup: dict[Path, EmbeddingArray] | None = None,
) -> StrategyComparisonReport:
    if overlay is None:
        # Offline acceptance comparison values deterministic CPU inference over
        # fragile OpenCL DNN startup paths more than peak throughput.
        os.environ.setdefault("GOD_MODE_DISABLE_OPENCL_DNN", "1")
    active_overlay: OverlayLike = overlay or cast(
        OverlayLike,
        GodModeOverlay(width=1280, height=720),
    )
    bundle = build_review_bundle(
        hard_positive_export=hard_positive_export,
        baseline_contacts_export=baseline_contacts_export,
        baseline_makeup_export=baseline_makeup_export,
        non_face_hard_negative_export=non_face_hard_negative_export,
        baseline_holdout_export=baseline_holdout_export,
    )
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    snapshot_path = snapshot_owner_embedding(
        owner_embedding_path=owner_embedding_path,
        snapshot_dir=snapshot_dir,
        timestamp=timestamp,
    )
    current_embeddings = normalize_owner_embeddings(
        cast(EmbeddingArray, np.load(owner_embedding_path))
    )

    sample_embeddings = collect_sample_embeddings(
        samples=(
            *bundle.append_candidates,
            *bundle.rebuild_sources,
            *bundle.non_face_owner_positives,
            *bundle.baseline_holdout,
            *bundle.guest_negative,
            *bundle.non_face_negative,
        ),
        overlay=active_overlay,
        embedding_lookup=embedding_lookup,
    )

    append_embeddings, selected_append_paths = build_append_candidate_embeddings(
        current_embeddings=current_embeddings,
        candidate_samples=bundle.append_candidates,
        negative_samples=bundle.negative_all,
        sample_embeddings=sample_embeddings,
        negative_penalty=negative_penalty,
        max_selected=max_selected,
    )
    append_candidate_path = snapshot_dir / f"owner_embedding_candidate_append_{timestamp}.npy"
    append_candidate_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(append_candidate_path, append_embeddings)

    rebuild_embeddings = build_rebuild_candidate_embeddings(
        source_samples=bundle.rebuild_sources,
        sample_embeddings=sample_embeddings,
    )
    rebuild_candidate_path = snapshot_dir / f"owner_embedding_candidate_rebuild_{timestamp}.npy"
    rebuild_candidate_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(rebuild_candidate_path, rebuild_embeddings)

    return StrategyComparisonReport(
        owner_embedding_path=owner_embedding_path,
        snapshot_path=snapshot_path,
        hard_positive_export=hard_positive_export,
        baseline_contacts_export=baseline_contacts_export,
        baseline_makeup_export=baseline_makeup_export,
        non_face_hard_negative_export=non_face_hard_negative_export,
        baseline_holdout_export=baseline_holdout_export,
        current=evaluate_strategy_report(
            owner_embeddings=current_embeddings,
            bundle=bundle,
            sample_embeddings=sample_embeddings,
            overlay=active_overlay,
            source_images=int(current_embeddings.shape[0]),
        ),
        append=evaluate_strategy_report(
            owner_embeddings=append_embeddings,
            bundle=bundle,
            sample_embeddings=sample_embeddings,
            overlay=active_overlay,
            source_images=len(bundle.append_candidates),
            added_embeddings=int(append_embeddings.shape[0] - current_embeddings.shape[0]),
            candidate_embedding_path=append_candidate_path,
            selected_source_paths=selected_append_paths,
        ),
        rebuild=evaluate_strategy_report(
            owner_embeddings=rebuild_embeddings,
            bundle=bundle,
            sample_embeddings=sample_embeddings,
            overlay=active_overlay,
            source_images=len(bundle.rebuild_sources),
            added_embeddings=int(rebuild_embeddings.shape[0]),
            candidate_embedding_path=rebuild_candidate_path,
            selected_source_paths=tuple(sample.source_image for sample in bundle.rebuild_sources),
        ),
    )


def collect_sample_embeddings(
    *,
    samples: tuple[ReviewedSample, ...],
    overlay: OverlayLike,
    embedding_lookup: dict[Path, EmbeddingArray] | None = None,
) -> dict[Path, EmbeddingArray]:
    embeddings: dict[Path, EmbeddingArray] = {}
    for sample in unique_samples_by_path(samples):
        if embedding_lookup is not None:
            embedding = embedding_lookup.get(sample.source_image)
            if embedding is None:
                continue
            embeddings[sample.source_image] = normalize_owner_embeddings(embedding)[0]
            continue
        frame = read_face_crop(sample.source_image)
        if frame is None:
            continue
        embedding = extract_crop_embedding(overlay, frame)
        if embedding is None:
            continue
        embeddings[sample.source_image] = normalize_owner_embeddings(embedding)[0]
    return embeddings


def build_append_candidate_embeddings(
    *,
    current_embeddings: EmbeddingArray,
    candidate_samples: tuple[ReviewedSample, ...],
    negative_samples: tuple[ReviewedSample, ...],
    sample_embeddings: dict[Path, EmbeddingArray],
    negative_penalty: float,
    max_selected: int | None,
) -> tuple[EmbeddingArray, tuple[Path, ...]]:
    candidate_paths, candidate_embeddings = embeddings_from_samples(
        samples=candidate_samples,
        sample_embeddings=sample_embeddings,
    )
    if len(candidate_paths) == 0:
        return current_embeddings.copy(), ()
    negative_paths, negative_embeddings = embeddings_from_samples(
        samples=negative_samples,
        sample_embeddings=sample_embeddings,
    )
    if len(negative_paths) == 0:
        return np.concatenate([current_embeddings, candidate_embeddings], axis=0), candidate_paths

    candidate_similarity = build_cosine_similarity_matrix(
        candidate_embeddings,
        candidate_embeddings,
    )
    negative_similarity = build_cosine_similarity_matrix(candidate_embeddings, negative_embeddings)
    current_positive_similarity = build_cosine_similarity_matrix(
        current_embeddings,
        candidate_embeddings,
    )
    current_negative_similarity = build_cosine_similarity_matrix(
        current_embeddings,
        negative_embeddings,
    )
    selection = greedy_select_false_negative_candidates(
        candidate_paths=candidate_paths,
        positive_candidate_scores=candidate_similarity,
        negative_candidate_scores=negative_similarity,
        positive_topk_values=build_topk_values(current_positive_similarity, topk=OWNER_TOPK),
        negative_topk_values=build_topk_values(current_negative_similarity, topk=OWNER_TOPK),
        threshold=OWNER_COSINE_THRESHOLD,
        negative_penalty=negative_penalty,
        max_selected=max_selected,
    )
    if not selection.selected_indices:
        return current_embeddings.copy(), ()
    selected_additions = candidate_embeddings[list(selection.selected_indices)]
    return (
        np.concatenate([current_embeddings, selected_additions], axis=0),
        selection.selected_paths,
    )


def build_rebuild_candidate_embeddings(
    *,
    source_samples: tuple[ReviewedSample, ...],
    sample_embeddings: dict[Path, EmbeddingArray],
) -> EmbeddingArray:
    source_paths, source_embeddings = embeddings_from_samples(
        samples=source_samples,
        sample_embeddings=sample_embeddings,
    )
    if len(source_paths) == 0:
        raise RuntimeError("no usable owner-positive embeddings available for rebuild strategy")
    return source_embeddings


def embeddings_from_samples(
    *,
    samples: tuple[ReviewedSample, ...],
    sample_embeddings: dict[Path, EmbeddingArray],
) -> tuple[tuple[Path, ...], EmbeddingArray]:
    ordered_paths: list[Path] = []
    ordered_embeddings: list[EmbeddingArray] = []
    for sample in unique_samples_by_path(samples):
        embedding = sample_embeddings.get(sample.source_image)
        if embedding is None:
            continue
        ordered_paths.append(sample.source_image)
        ordered_embeddings.append(embedding)
    if not ordered_embeddings:
        return (), cast(EmbeddingArray, np.zeros((0, 1, 128), dtype=np.float32))
    return tuple(ordered_paths), cast(EmbeddingArray, np.stack(ordered_embeddings, axis=0))


def evaluate_strategy_report(
    *,
    owner_embeddings: EmbeddingArray,
    bundle: ReviewBundle,
    sample_embeddings: dict[Path, EmbeddingArray],
    overlay: OverlayLike,
    source_images: int,
    added_embeddings: int = 0,
    candidate_embedding_path: Path | None = None,
    selected_source_paths: tuple[Path, ...] = (),
) -> StrategyEvaluationReport:
    classifier = build_classifier(overlay=overlay, owner_embeddings=owner_embeddings)
    return StrategyEvaluationReport(
        bank_size=int(owner_embeddings.shape[0]),
        source_images=source_images,
        added_embeddings=added_embeddings,
        hard_positive_glasses=evaluate_review_samples(
            samples=bundle.hard_positive_glasses,
            sample_embeddings=sample_embeddings,
            classify_embedding=classifier,
        ),
        baseline_contacts=evaluate_review_samples(
            samples=bundle.baseline_contacts,
            sample_embeddings=sample_embeddings,
            classify_embedding=classifier,
        ),
        baseline_makeup=evaluate_review_samples(
            samples=bundle.baseline_makeup,
            sample_embeddings=sample_embeddings,
            classify_embedding=classifier,
        ),
        non_face_owner_positives=evaluate_review_samples(
            samples=bundle.non_face_owner_positives,
            sample_embeddings=sample_embeddings,
            classify_embedding=classifier,
        ),
        baseline_holdout=evaluate_review_samples(
            samples=bundle.baseline_holdout,
            sample_embeddings=sample_embeddings,
            classify_embedding=classifier,
        ),
        guest_negative=evaluate_review_samples(
            samples=bundle.guest_negative,
            sample_embeddings=sample_embeddings,
            classify_embedding=classifier,
        ),
        non_face_negative=evaluate_review_samples(
            samples=bundle.non_face_negative,
            sample_embeddings=sample_embeddings,
            classify_embedding=classifier,
        ),
        negative_all=evaluate_review_samples(
            samples=bundle.negative_all,
            sample_embeddings=sample_embeddings,
            classify_embedding=classifier,
        ),
        candidate_embedding_path=candidate_embedding_path,
        selected_source_paths=selected_source_paths,
    )


def build_classifier(
    *,
    overlay: OverlayLike,
    owner_embeddings: EmbeddingArray,
) -> Callable[[Path, EmbeddingArray], tuple[str, float]]:
    def _classify(path: Path, embedding: EmbeddingArray) -> tuple[str, float]:
        del path
        return classify_owner_embedding(
            recognizer=overlay._recognizer,
            owner_embeddings=owner_embeddings,
            embedding=embedding,
            face_confidence=1.0,
        )

    return _classify


def evaluate_review_samples(
    *,
    samples: tuple[ReviewedSample, ...],
    sample_embeddings: dict[Path, EmbeddingArray],
    classify_embedding: Callable[[Path, EmbeddingArray], tuple[str, float]],
) -> DatasetEvaluation:
    total_files = len(samples)
    usable_embeddings = 0
    skipped_files = 0
    owner_predictions = 0
    subject_predictions = 0
    scores: list[float] = []
    for sample in samples:
        embedding = sample_embeddings.get(sample.source_image)
        if embedding is None:
            skipped_files += 1
            continue
        usable_embeddings += 1
        label, score = classify_embedding(sample.source_image, embedding)
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


def format_positive_evaluation(name: str, evaluation: DatasetEvaluation) -> str:
    usable = max(evaluation.usable_embeddings, 1)
    recall = evaluation.owner_predictions / usable
    return (
        f"{name}: total={evaluation.total_files} usable={evaluation.usable_embeddings} "
        f"owner={evaluation.owner_predictions} subject={evaluation.subject_predictions} "
        f"recall={recall:.4f} mean_score={evaluation.mean_score:.4f}"
    )


def format_negative_evaluation(name: str, evaluation: DatasetEvaluation) -> str:
    usable = max(evaluation.usable_embeddings, 1)
    false_positive_rate = evaluation.owner_predictions / usable
    return (
        f"{name}: total={evaluation.total_files} usable={evaluation.usable_embeddings} "
        f"false_owner={evaluation.owner_predictions} kept_subject={evaluation.subject_predictions} "
        f"false_positive_rate={false_positive_rate:.4f} mean_score={evaluation.mean_score:.4f}"
    )


def format_strategy(name: str, report: StrategyEvaluationReport) -> list[str]:
    lines = [
        f"[{name}] bank_size={report.bank_size} source_images={report.source_images} "
        f"added_embeddings={report.added_embeddings}",
        format_positive_evaluation("hard_positive_glasses", report.hard_positive_glasses),
        format_positive_evaluation("baseline_contacts", report.baseline_contacts),
        format_positive_evaluation("baseline_makeup", report.baseline_makeup),
        format_positive_evaluation("non_face_owner_positives", report.non_face_owner_positives),
        format_positive_evaluation("baseline_holdout", report.baseline_holdout),
        format_negative_evaluation("guest_negative", report.guest_negative),
        format_negative_evaluation("non_face_negative", report.non_face_negative),
        format_negative_evaluation("negative_all", report.negative_all),
    ]
    if report.candidate_embedding_path is not None:
        lines.append(f"candidate_embedding={report.candidate_embedding_path}")
    if report.selected_source_paths:
        lines.append(f"selected_sources={len(report.selected_source_paths)}")
    return lines


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    hard_positive_export = (
        Path(args.hard_positive_export)
        if args.hard_positive_export is not None
        else resolve_latest_export_json(DEFAULT_HARD_POSITIVE_BACKUP_ROOT)
    )
    baseline_contacts_export = (
        Path(args.baseline_contacts_export)
        if args.baseline_contacts_export is not None
        else resolve_latest_export_json(DEFAULT_BASELINE_CONTACTS_BACKUP_ROOT)
    )
    baseline_makeup_export = (
        Path(args.baseline_makeup_export)
        if args.baseline_makeup_export is not None
        else resolve_latest_export_json(DEFAULT_BASELINE_MAKEUP_BACKUP_ROOT)
    )
    non_face_hard_negative_export = (
        Path(args.non_face_hard_negative_export)
        if args.non_face_hard_negative_export is not None
        else resolve_latest_export_json(DEFAULT_NON_FACE_HARD_NEGATIVE_BACKUP_ROOT)
    )
    baseline_holdout_export = (
        Path(args.baseline_holdout_export)
        if args.baseline_holdout_export is not None
        else resolve_latest_export_json(DEFAULT_BASELINE_HOLDOUT_BACKUP_ROOT)
    )
    report = compare_owner_embedding_strategies(
        owner_embedding_path=Path(args.owner_embedding_path),
        hard_positive_export=hard_positive_export,
        baseline_contacts_export=baseline_contacts_export,
        baseline_makeup_export=baseline_makeup_export,
        non_face_hard_negative_export=non_face_hard_negative_export,
        baseline_holdout_export=baseline_holdout_export,
        snapshot_dir=Path(args.snapshot_dir),
        negative_penalty=float(args.negative_penalty),
        max_selected=None if args.max_selected is None else int(args.max_selected),
    )
    print(f"owner_embedding={report.owner_embedding_path}")
    print(f"snapshot={report.snapshot_path}")
    print(f"hard_positive_export={report.hard_positive_export}")
    print(f"baseline_contacts_export={report.baseline_contacts_export}")
    print(f"baseline_makeup_export={report.baseline_makeup_export}")
    print(f"non_face_hard_negative_export={report.non_face_hard_negative_export}")
    print(f"baseline_holdout_export={report.baseline_holdout_export}")
    for line in format_strategy("current", report.current):
        print(line)
    for line in format_strategy("append", report.append):
        print(line)
    for line in format_strategy("rebuild", report.rebuild):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Matrix runner for owner embedding update experiments."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import cast

import numpy as np

from .compare_owner_embedding_strategies import (
    DEFAULT_BASELINE_CONTACTS_BACKUP_ROOT,
    DEFAULT_BASELINE_HOLDOUT_BACKUP_ROOT,
    DEFAULT_BASELINE_MAKEUP_BACKUP_ROOT,
    DEFAULT_DARK_ROOM_MORNING_BACKUP_ROOT,
    DEFAULT_HARD_POSITIVE_BACKUP_ROOT,
    DEFAULT_NON_FACE_HARD_NEGATIVE_BACKUP_ROOT,
    DEFAULT_WEAK_BASELINE_MAKEUP_ROOT,
    DEFAULT_WEAK_BASELINE_NON_MAKEUP_ROOT,
    ReviewBundle,
    ReviewedSample,
    StrategyEvaluationReport,
    build_review_bundle,
    collect_sample_embeddings,
    embeddings_from_samples,
    format_negative_evaluation,
    format_positive_evaluation,
    resolve_latest_export_json,
)
from .enroll_owner import DEFAULT_OWNER_EMBED_PATH
from .owner_policy import OWNER_COSINE_THRESHOLD, OWNER_TOPK
from .retrain_owner_embedding import (
    DEFAULT_SNAPSHOT_DIR,
    DatasetEvaluation,
    EmbeddingArray,
    OverlayLike,
    build_cosine_similarity_matrix,
    build_topk_values,
    normalize_owner_embeddings,
    snapshot_owner_embedding,
)


@dataclass(frozen=True, slots=True)
class ExperimentSourceGroup:
    key: str
    samples: tuple[ReviewedSample, ...]


@dataclass(frozen=True, slots=True)
class ExperimentStrategy:
    key: str
    mode: str
    negative_penalty: float | None = None
    max_selected: int | None = None


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    source_group_key: str
    strategy_key: str
    bank_size: int
    source_images: int
    added_embeddings: int
    selected_source_paths: tuple[Path, ...]
    candidate_embedding_path: Path
    hard_positive_gain: int
    baseline_contacts_gain: int
    baseline_makeup_gain: int
    non_face_owner_gain: int
    baseline_holdout_gain: int
    dark_room_morning_gain: int
    weak_non_makeup_owner_gain: int
    weak_non_makeup_false_negative_gain: int
    weak_makeup_owner_gain: int
    weak_makeup_false_negative_gain: int
    guest_negative_delta: int
    non_face_negative_delta: int
    negative_all_delta: int
    report: StrategyEvaluationReport


@dataclass(frozen=True, slots=True)
class ExperimentMatrixReport:
    owner_embedding_path: Path
    snapshot_path: Path
    output_dir: Path
    summary_json_path: Path
    current: StrategyEvaluationReport
    results: tuple[ExperimentResult, ...]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run source-group x strategy experiments for owner embedding updates"
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
        "--dark-room-morning-export",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--weak-baseline-non-makeup-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--weak-baseline-makeup-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=DEFAULT_SNAPSHOT_DIR,
    )
    parser.add_argument(
        "--greedy-penalty",
        type=float,
        action="append",
        default=None,
        help="Append-greedy penalties to include in the matrix (default: 3.0 and 1.5)",
    )
    parser.add_argument(
        "--max-selected",
        type=int,
        default=None,
    )
    return parser


def build_default_source_groups(bundle: ReviewBundle) -> tuple[ExperimentSourceGroup, ...]:
    project_groups = tuple(
        (name, samples)
        for name, samples in (
            ("hard_positive_glasses", bundle.hard_positive_glasses),
            ("baseline_contacts", bundle.baseline_contacts),
            ("baseline_makeup", bundle.baseline_makeup),
            ("dark_room_morning", bundle.dark_room_morning),
            ("non_face_owner_positives", bundle.non_face_owner_positives),
            ("weak_non_makeup_false_negative", bundle.weak_non_makeup_false_negative),
            ("weak_makeup_false_negative", bundle.weak_makeup_false_negative),
        )
        if samples
    )
    groups: list[ExperimentSourceGroup] = []
    for width in range(1, len(project_groups) + 1):
        for subset in combinations(project_groups, width):
            key = "+".join(name for name, _samples in subset)
            merged = tuple(sample for _name, samples in subset for sample in samples)
            groups.append(
                ExperimentSourceGroup(
                    key=key,
                    samples=merged,
                )
            )
    return tuple(groups)


def build_default_strategies(
    *,
    max_selected: int | None = None,
    greedy_penalties: tuple[float, ...] = (3.0, 1.5),
) -> tuple[ExperimentStrategy, ...]:
    strategies: list[ExperimentStrategy] = []
    for penalty in greedy_penalties:
        strategies.append(
            ExperimentStrategy(
                key=f"append_greedy_p{penalty:.1f}",
                mode="append_greedy",
                negative_penalty=penalty,
                max_selected=max_selected,
            )
        )
    strategies.append(ExperimentStrategy(key="append_full", mode="append_full"))
    strategies.append(ExperimentStrategy(key="rebuild", mode="rebuild"))
    return tuple(strategies)


def run_owner_embedding_experiment_matrix(
    *,
    owner_embedding_path: Path,
    hard_positive_export: Path,
    baseline_contacts_export: Path,
    baseline_makeup_export: Path,
    non_face_hard_negative_export: Path | None = None,
    baseline_holdout_export: Path | None = None,
    dark_room_morning_export: Path | None = None,
    weak_baseline_non_makeup_root: Path | None = None,
    weak_baseline_makeup_root: Path | None = None,
    snapshot_dir: Path,
    overlay: OverlayLike | None = None,
    embedding_lookup: dict[Path, EmbeddingArray] | None = None,
    source_groups: tuple[ExperimentSourceGroup, ...] | None = None,
    strategies: tuple[ExperimentStrategy, ...] | None = None,
) -> ExperimentMatrixReport:
    bundle = build_review_bundle(
        hard_positive_export=hard_positive_export,
        baseline_contacts_export=baseline_contacts_export,
        baseline_makeup_export=baseline_makeup_export,
        non_face_hard_negative_export=non_face_hard_negative_export,
        baseline_holdout_export=baseline_holdout_export,
        dark_room_morning_export=dark_room_morning_export,
        weak_baseline_non_makeup_root=weak_baseline_non_makeup_root,
        weak_baseline_makeup_root=weak_baseline_makeup_root,
    )
    if overlay is not None:
        active_overlay = overlay
    elif embedding_lookup is not None:
        active_overlay = None
    else:
        os.environ.setdefault("GOD_MODE_DISABLE_OPENCL_DNN", "1")
        from .overlay import GodModeOverlay

        active_overlay = GodModeOverlay(width=1280, height=720)
    sample_embeddings = collect_sample_embeddings(
        samples=(
            *bundle.hard_positive_glasses,
            *bundle.baseline_contacts,
            *bundle.baseline_makeup,
            *bundle.non_face_owner_positives,
            *bundle.baseline_holdout,
            *bundle.dark_room_morning,
            *bundle.weak_non_makeup_owner_raw,
            *bundle.weak_non_makeup_false_negative,
            *bundle.weak_makeup_owner_raw,
            *bundle.weak_makeup_false_negative,
            *bundle.guest_negative,
            *bundle.non_face_negative,
        ),
        overlay=active_overlay if active_overlay is not None else cast(OverlayLike, object()),
        embedding_lookup=embedding_lookup,
    )
    current_embeddings = normalize_owner_embeddings(np.load(owner_embedding_path))
    if source_groups is None:
        source_groups = build_default_source_groups(bundle)
    if strategies is None:
        strategies = build_default_strategies()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    snapshot_path = snapshot_owner_embedding(
        owner_embedding_path=owner_embedding_path,
        snapshot_dir=snapshot_dir,
        timestamp=timestamp,
    )
    output_dir = snapshot_dir / f"owner_embedding_experiment_matrix_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    current_report = evaluate_strategy_report_fast(
        owner_embeddings=current_embeddings,
        bundle=bundle,
        sample_embeddings=sample_embeddings,
        source_images=int(current_embeddings.shape[0]),
        added_embeddings=0,
        candidate_embedding_path=None,
        selected_source_paths=(),
    )

    results: list[ExperimentResult] = []
    for source_group in source_groups:
        for strategy in strategies:
            candidate_embeddings, selected_paths = build_strategy_candidate_embeddings(
                current_embeddings=current_embeddings,
                source_group=source_group,
                strategy=strategy,
                negative_samples=bundle.negative_all,
                sample_embeddings=sample_embeddings,
            )
            candidate_path = output_dir / (
                f"{source_group.key}__{strategy.key}".replace("+", "__") + ".npy"
            )
            np.save(candidate_path, candidate_embeddings)
            evaluation = evaluate_strategy_report_fast(
                owner_embeddings=candidate_embeddings,
                bundle=bundle,
                sample_embeddings=sample_embeddings,
                source_images=len(source_group.samples),
                added_embeddings=int(candidate_embeddings.shape[0] - current_embeddings.shape[0])
                if strategy.mode != "rebuild"
                else int(candidate_embeddings.shape[0]),
                candidate_embedding_path=candidate_path,
                selected_source_paths=selected_paths,
            )
            results.append(
                build_experiment_result(
                    source_group_key=source_group.key,
                    strategy_key=strategy.key,
                    report=evaluation,
                    current=current_report,
                )
            )

    ranked_results = tuple(sorted(results, key=rank_experiment_result))
    summary_json_path = output_dir / "summary.json"
    summary_json_path.write_text(
        json.dumps(
            {
                "owner_embedding_path": str(owner_embedding_path),
                "snapshot_path": str(snapshot_path),
                "current": strategy_report_to_dict(current_report),
                "results": [experiment_result_to_dict(result) for result in ranked_results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return ExperimentMatrixReport(
        owner_embedding_path=owner_embedding_path,
        snapshot_path=snapshot_path,
        output_dir=output_dir,
        summary_json_path=summary_json_path,
        current=current_report,
        results=ranked_results,
    )


def build_strategy_candidate_embeddings(
    *,
    current_embeddings: EmbeddingArray,
    source_group: ExperimentSourceGroup,
    strategy: ExperimentStrategy,
    negative_samples: tuple[ReviewedSample, ...],
    sample_embeddings: dict[Path, EmbeddingArray],
) -> tuple[EmbeddingArray, tuple[Path, ...]]:
    source_paths, source_embeddings = embeddings_from_samples(
        samples=source_group.samples,
        sample_embeddings=sample_embeddings,
    )
    if strategy.mode == "append_full":
        if len(source_paths) == 0:
            return current_embeddings.copy(), ()
        return np.concatenate([current_embeddings, source_embeddings], axis=0), source_paths
    if strategy.mode == "rebuild":
        if len(source_paths) == 0:
            raise RuntimeError(f"no usable embeddings for rebuild source group {source_group.key}")
        return source_embeddings, source_paths
    if strategy.mode != "append_greedy":
        raise ValueError(f"unsupported strategy mode: {strategy.mode}")
    if len(source_paths) == 0:
        return current_embeddings.copy(), ()
    negative_paths, negative_embeddings = embeddings_from_samples(
        samples=negative_samples,
        sample_embeddings=sample_embeddings,
    )
    if len(negative_paths) == 0:
        return np.concatenate([current_embeddings, source_embeddings], axis=0), source_paths
    candidate_similarity = build_cosine_similarity_matrix(source_embeddings, source_embeddings)
    negative_similarity = build_cosine_similarity_matrix(source_embeddings, negative_embeddings)
    current_positive_similarity = build_cosine_similarity_matrix(
        current_embeddings,
        source_embeddings,
    )
    current_negative_similarity = build_cosine_similarity_matrix(
        current_embeddings,
        negative_embeddings,
    )
    topk_positive = build_topk_values(current_positive_similarity, topk=OWNER_TOPK)
    topk_negative = build_topk_values(current_negative_similarity, topk=OWNER_TOPK)
    selected_indices = greedy_select_indices(
        positive_candidate_scores=candidate_similarity,
        negative_candidate_scores=negative_similarity,
        positive_topk_values=topk_positive,
        negative_topk_values=topk_negative,
        threshold=OWNER_COSINE_THRESHOLD,
        negative_penalty=1.0 if strategy.negative_penalty is None else strategy.negative_penalty,
        max_selected=strategy.max_selected,
    )
    if not selected_indices:
        return current_embeddings.copy(), ()
    selected_embeddings = source_embeddings[list(selected_indices)]
    selected_paths = tuple(source_paths[index] for index in selected_indices)
    return np.concatenate([current_embeddings, selected_embeddings], axis=0), selected_paths


def greedy_select_indices(
    *,
    positive_candidate_scores: np.ndarray,
    negative_candidate_scores: np.ndarray,
    positive_topk_values: np.ndarray,
    negative_topk_values: np.ndarray,
    threshold: float,
    negative_penalty: float,
    max_selected: int | None,
) -> tuple[int, ...]:
    from .retrain_owner_embedding import (
        GreedySelectionResult,
        greedy_select_false_negative_candidates,
    )

    candidate_paths = [Path(str(index)) for index in range(positive_candidate_scores.shape[0])]
    result: GreedySelectionResult = greedy_select_false_negative_candidates(
        candidate_paths=candidate_paths,
        positive_candidate_scores=positive_candidate_scores,
        negative_candidate_scores=negative_candidate_scores,
        positive_topk_values=positive_topk_values,
        negative_topk_values=negative_topk_values,
        threshold=threshold,
        negative_penalty=negative_penalty,
        max_selected=max_selected,
    )
    return tuple(int(path.name) for path in result.selected_paths)


def evaluate_strategy_report_fast(
    *,
    owner_embeddings: EmbeddingArray,
    bundle: ReviewBundle,
    sample_embeddings: dict[Path, EmbeddingArray],
    source_images: int,
    added_embeddings: int,
    candidate_embedding_path: Path | None,
    selected_source_paths: tuple[Path, ...],
) -> StrategyEvaluationReport:
    return StrategyEvaluationReport(
        bank_size=int(owner_embeddings.shape[0]),
        source_images=source_images,
        added_embeddings=added_embeddings,
        hard_positive_glasses=evaluate_review_samples_fast(
            samples=bundle.hard_positive_glasses,
            sample_embeddings=sample_embeddings,
            owner_embeddings=owner_embeddings,
        ),
        baseline_contacts=evaluate_review_samples_fast(
            samples=bundle.baseline_contacts,
            sample_embeddings=sample_embeddings,
            owner_embeddings=owner_embeddings,
        ),
        baseline_makeup=evaluate_review_samples_fast(
            samples=bundle.baseline_makeup,
            sample_embeddings=sample_embeddings,
            owner_embeddings=owner_embeddings,
        ),
        non_face_owner_positives=evaluate_review_samples_fast(
            samples=bundle.non_face_owner_positives,
            sample_embeddings=sample_embeddings,
            owner_embeddings=owner_embeddings,
        ),
        baseline_holdout=evaluate_review_samples_fast(
            samples=bundle.baseline_holdout,
            sample_embeddings=sample_embeddings,
            owner_embeddings=owner_embeddings,
        ),
        dark_room_morning=evaluate_review_samples_fast(
            samples=bundle.dark_room_morning,
            sample_embeddings=sample_embeddings,
            owner_embeddings=owner_embeddings,
        ),
        weak_non_makeup_owner_raw=evaluate_review_samples_fast(
            samples=bundle.weak_non_makeup_owner_raw,
            sample_embeddings=sample_embeddings,
            owner_embeddings=owner_embeddings,
        ),
        weak_non_makeup_false_negative=evaluate_review_samples_fast(
            samples=bundle.weak_non_makeup_false_negative,
            sample_embeddings=sample_embeddings,
            owner_embeddings=owner_embeddings,
        ),
        weak_makeup_owner_raw=evaluate_review_samples_fast(
            samples=bundle.weak_makeup_owner_raw,
            sample_embeddings=sample_embeddings,
            owner_embeddings=owner_embeddings,
        ),
        weak_makeup_false_negative=evaluate_review_samples_fast(
            samples=bundle.weak_makeup_false_negative,
            sample_embeddings=sample_embeddings,
            owner_embeddings=owner_embeddings,
        ),
        guest_negative=evaluate_review_samples_fast(
            samples=bundle.guest_negative,
            sample_embeddings=sample_embeddings,
            owner_embeddings=owner_embeddings,
        ),
        non_face_negative=evaluate_review_samples_fast(
            samples=bundle.non_face_negative,
            sample_embeddings=sample_embeddings,
            owner_embeddings=owner_embeddings,
        ),
        negative_all=evaluate_review_samples_fast(
            samples=bundle.negative_all,
            sample_embeddings=sample_embeddings,
            owner_embeddings=owner_embeddings,
        ),
        candidate_embedding_path=candidate_embedding_path,
        selected_source_paths=selected_source_paths,
    )


def evaluate_review_samples_fast(
    *,
    samples: tuple[ReviewedSample, ...],
    sample_embeddings: dict[Path, EmbeddingArray],
    owner_embeddings: EmbeddingArray,
) -> DatasetEvaluation:
    total_files = len(samples)
    sample_paths, sample_bank = embeddings_from_samples(
        samples=samples,
        sample_embeddings=sample_embeddings,
    )
    usable_embeddings = len(sample_paths)
    skipped_files = total_files - usable_embeddings
    if usable_embeddings == 0:
        return DatasetEvaluation(
            total_files=total_files,
            usable_embeddings=0,
            skipped_files=skipped_files,
            owner_predictions=0,
            subject_predictions=0,
            mean_score=0.0,
        )
    similarities = build_cosine_similarity_matrix(owner_embeddings, sample_bank)
    topk_values = build_topk_values(similarities, topk=OWNER_TOPK)
    scores = topk_values.mean(axis=1)
    owner_predictions = int(np.count_nonzero(scores >= OWNER_COSINE_THRESHOLD))
    subject_predictions = usable_embeddings - owner_predictions
    return DatasetEvaluation(
        total_files=total_files,
        usable_embeddings=usable_embeddings,
        skipped_files=skipped_files,
        owner_predictions=owner_predictions,
        subject_predictions=subject_predictions,
        mean_score=float(np.mean(scores)),
    )


def build_experiment_result(
    *,
    source_group_key: str,
    strategy_key: str,
    report: StrategyEvaluationReport,
    current: StrategyEvaluationReport,
) -> ExperimentResult:
    return ExperimentResult(
        source_group_key=source_group_key,
        strategy_key=strategy_key,
        bank_size=report.bank_size,
        source_images=report.source_images,
        added_embeddings=report.added_embeddings,
        selected_source_paths=report.selected_source_paths,
        candidate_embedding_path=report.candidate_embedding_path
        if report.candidate_embedding_path is not None
        else Path(""),
        hard_positive_gain=report.hard_positive_glasses.owner_predictions
        - current.hard_positive_glasses.owner_predictions,
        baseline_contacts_gain=report.baseline_contacts.owner_predictions
        - current.baseline_contacts.owner_predictions,
        baseline_makeup_gain=report.baseline_makeup.owner_predictions
        - current.baseline_makeup.owner_predictions,
        non_face_owner_gain=report.non_face_owner_positives.owner_predictions
        - current.non_face_owner_positives.owner_predictions,
        baseline_holdout_gain=report.baseline_holdout.owner_predictions
        - current.baseline_holdout.owner_predictions,
        dark_room_morning_gain=report.dark_room_morning.owner_predictions
        - current.dark_room_morning.owner_predictions,
        weak_non_makeup_owner_gain=report.weak_non_makeup_owner_raw.owner_predictions
        - current.weak_non_makeup_owner_raw.owner_predictions,
        weak_non_makeup_false_negative_gain=report.weak_non_makeup_false_negative.owner_predictions
        - current.weak_non_makeup_false_negative.owner_predictions,
        weak_makeup_owner_gain=report.weak_makeup_owner_raw.owner_predictions
        - current.weak_makeup_owner_raw.owner_predictions,
        weak_makeup_false_negative_gain=report.weak_makeup_false_negative.owner_predictions
        - current.weak_makeup_false_negative.owner_predictions,
        guest_negative_delta=report.guest_negative.owner_predictions
        - current.guest_negative.owner_predictions,
        non_face_negative_delta=report.non_face_negative.owner_predictions
        - current.non_face_negative.owner_predictions,
        negative_all_delta=report.negative_all.owner_predictions
        - current.negative_all.owner_predictions,
        report=report,
    )


def rank_experiment_result(result: ExperimentResult) -> tuple[int, int, int, int, int, int]:
    safe_rank = 0 if result.guest_negative_delta <= 0 and result.non_face_negative_delta <= 0 else 1
    return (
        safe_rank,
        result.guest_negative_delta,
        result.non_face_negative_delta,
        result.negative_all_delta,
        -(
            result.hard_positive_gain
            + result.baseline_contacts_gain
            + result.baseline_makeup_gain
            + result.non_face_owner_gain
            + result.baseline_holdout_gain
            + result.dark_room_morning_gain
            + result.weak_non_makeup_owner_gain
            + result.weak_non_makeup_false_negative_gain
            + result.weak_makeup_owner_gain
            + result.weak_makeup_false_negative_gain
        ),
        result.added_embeddings,
    )


def strategy_report_to_dict(report: StrategyEvaluationReport) -> dict[str, object]:
    return {
        "bank_size": report.bank_size,
        "source_images": report.source_images,
        "added_embeddings": report.added_embeddings,
        "hard_positive_glasses": dataset_evaluation_to_dict(report.hard_positive_glasses),
        "baseline_contacts": dataset_evaluation_to_dict(report.baseline_contacts),
        "baseline_makeup": dataset_evaluation_to_dict(report.baseline_makeup),
        "non_face_owner_positives": dataset_evaluation_to_dict(report.non_face_owner_positives),
        "baseline_holdout": dataset_evaluation_to_dict(report.baseline_holdout),
        "dark_room_morning": dataset_evaluation_to_dict(report.dark_room_morning),
        "weak_non_makeup_owner_raw": dataset_evaluation_to_dict(report.weak_non_makeup_owner_raw),
        "weak_non_makeup_false_negative": dataset_evaluation_to_dict(
            report.weak_non_makeup_false_negative
        ),
        "weak_makeup_owner_raw": dataset_evaluation_to_dict(report.weak_makeup_owner_raw),
        "weak_makeup_false_negative": dataset_evaluation_to_dict(
            report.weak_makeup_false_negative
        ),
        "guest_negative": dataset_evaluation_to_dict(report.guest_negative),
        "non_face_negative": dataset_evaluation_to_dict(report.non_face_negative),
        "negative_all": dataset_evaluation_to_dict(report.negative_all),
        "candidate_embedding_path": str(report.candidate_embedding_path)
        if report.candidate_embedding_path is not None
        else None,
        "selected_source_paths": [str(path) for path in report.selected_source_paths],
    }


def experiment_result_to_dict(result: ExperimentResult) -> dict[str, object]:
    return {
        "source_group_key": result.source_group_key,
        "strategy_key": result.strategy_key,
        "bank_size": result.bank_size,
        "source_images": result.source_images,
        "added_embeddings": result.added_embeddings,
        "selected_source_paths": [str(path) for path in result.selected_source_paths],
        "candidate_embedding_path": str(result.candidate_embedding_path),
        "hard_positive_gain": result.hard_positive_gain,
        "baseline_contacts_gain": result.baseline_contacts_gain,
        "baseline_makeup_gain": result.baseline_makeup_gain,
        "non_face_owner_gain": result.non_face_owner_gain,
        "baseline_holdout_gain": result.baseline_holdout_gain,
        "dark_room_morning_gain": result.dark_room_morning_gain,
        "weak_non_makeup_owner_gain": result.weak_non_makeup_owner_gain,
        "weak_non_makeup_false_negative_gain": result.weak_non_makeup_false_negative_gain,
        "weak_makeup_owner_gain": result.weak_makeup_owner_gain,
        "weak_makeup_false_negative_gain": result.weak_makeup_false_negative_gain,
        "guest_negative_delta": result.guest_negative_delta,
        "non_face_negative_delta": result.non_face_negative_delta,
        "negative_all_delta": result.negative_all_delta,
        "report": strategy_report_to_dict(result.report),
    }


def dataset_evaluation_to_dict(evaluation: DatasetEvaluation) -> dict[str, object]:
    return {
        "total_files": evaluation.total_files,
        "usable_embeddings": evaluation.usable_embeddings,
        "skipped_files": evaluation.skipped_files,
        "owner_predictions": evaluation.owner_predictions,
        "subject_predictions": evaluation.subject_predictions,
        "mean_score": evaluation.mean_score,
    }


def format_experiment_result(result: ExperimentResult) -> list[str]:
    return [
        (
            f"[{result.source_group_key} / {result.strategy_key}] bank_size={result.bank_size} "
            f"added={result.added_embeddings} hard_gain={result.hard_positive_gain} "
            f"contacts_gain={result.baseline_contacts_gain} "
            f"makeup_gain={result.baseline_makeup_gain} "
            f"nonface_owner_gain={result.non_face_owner_gain} "
            f"holdout_gain={result.baseline_holdout_gain} "
            f"dark_room_gain={result.dark_room_morning_gain} "
            f"weak_non_makeup_owner_gain={result.weak_non_makeup_owner_gain} "
            f"weak_non_makeup_fn_gain={result.weak_non_makeup_false_negative_gain} "
            f"weak_makeup_owner_gain={result.weak_makeup_owner_gain} "
            f"weak_makeup_fn_gain={result.weak_makeup_false_negative_gain} "
            f"guest_delta={result.guest_negative_delta} "
            f"nonface_delta={result.non_face_negative_delta} "
            f"negative_all_delta={result.negative_all_delta}"
        ),
        format_positive_evaluation("hard_positive_glasses", result.report.hard_positive_glasses),
        format_positive_evaluation("baseline_contacts", result.report.baseline_contacts),
        format_positive_evaluation("baseline_makeup", result.report.baseline_makeup),
        format_positive_evaluation(
            "non_face_owner_positives",
            result.report.non_face_owner_positives,
        ),
        format_positive_evaluation("baseline_holdout", result.report.baseline_holdout),
        format_positive_evaluation("dark_room_morning", result.report.dark_room_morning),
        format_positive_evaluation(
            "weak_non_makeup_owner_raw",
            result.report.weak_non_makeup_owner_raw,
        ),
        format_positive_evaluation(
            "weak_non_makeup_false_negative",
            result.report.weak_non_makeup_false_negative,
        ),
        format_positive_evaluation(
            "weak_makeup_owner_raw",
            result.report.weak_makeup_owner_raw,
        ),
        format_positive_evaluation(
            "weak_makeup_false_negative",
            result.report.weak_makeup_false_negative,
        ),
        format_negative_evaluation("guest_negative", result.report.guest_negative),
        format_negative_evaluation("non_face_negative", result.report.non_face_negative),
        format_negative_evaluation("negative_all", result.report.negative_all),
    ]


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
    dark_room_morning_export = (
        Path(args.dark_room_morning_export)
        if args.dark_room_morning_export is not None
        else resolve_latest_export_json(DEFAULT_DARK_ROOM_MORNING_BACKUP_ROOT)
    )
    weak_baseline_non_makeup_root = (
        Path(args.weak_baseline_non_makeup_root)
        if args.weak_baseline_non_makeup_root is not None
        else DEFAULT_WEAK_BASELINE_NON_MAKEUP_ROOT
    )
    weak_baseline_makeup_root = (
        Path(args.weak_baseline_makeup_root)
        if args.weak_baseline_makeup_root is not None
        else DEFAULT_WEAK_BASELINE_MAKEUP_ROOT
    )
    penalties = tuple(args.greedy_penalty) if args.greedy_penalty else (3.0, 1.5)
    report = run_owner_embedding_experiment_matrix(
        owner_embedding_path=Path(args.owner_embedding_path),
        hard_positive_export=hard_positive_export,
        baseline_contacts_export=baseline_contacts_export,
        baseline_makeup_export=baseline_makeup_export,
        non_face_hard_negative_export=non_face_hard_negative_export,
        baseline_holdout_export=baseline_holdout_export,
        dark_room_morning_export=dark_room_morning_export,
        weak_baseline_non_makeup_root=weak_baseline_non_makeup_root,
        weak_baseline_makeup_root=weak_baseline_makeup_root,
        snapshot_dir=Path(args.snapshot_dir),
        strategies=build_default_strategies(
            max_selected=None if args.max_selected is None else int(args.max_selected),
            greedy_penalties=penalties,
        ),
    )
    print(f"owner_embedding={report.owner_embedding_path}")
    print(f"snapshot={report.snapshot_path}")
    print(f"output_dir={report.output_dir}")
    print(f"summary_json={report.summary_json_path}")
    print("[current]")
    print(format_positive_evaluation("hard_positive_glasses", report.current.hard_positive_glasses))
    print(format_positive_evaluation("baseline_contacts", report.current.baseline_contacts))
    print(format_positive_evaluation("baseline_makeup", report.current.baseline_makeup))
    print(
        format_positive_evaluation(
            "non_face_owner_positives",
            report.current.non_face_owner_positives,
        )
    )
    print(format_positive_evaluation("baseline_holdout", report.current.baseline_holdout))
    print(format_positive_evaluation("dark_room_morning", report.current.dark_room_morning))
    print(
        format_positive_evaluation(
            "weak_non_makeup_owner_raw",
            report.current.weak_non_makeup_owner_raw,
        )
    )
    print(
        format_positive_evaluation(
            "weak_non_makeup_false_negative",
            report.current.weak_non_makeup_false_negative,
        )
    )
    print(
        format_positive_evaluation(
            "weak_makeup_owner_raw",
            report.current.weak_makeup_owner_raw,
        )
    )
    print(
        format_positive_evaluation(
            "weak_makeup_false_negative",
            report.current.weak_makeup_false_negative,
        )
    )
    print(format_negative_evaluation("guest_negative", report.current.guest_negative))
    print(format_negative_evaluation("non_face_negative", report.current.non_face_negative))
    print(format_negative_evaluation("negative_all", report.current.negative_all))
    for result in report.results[:10]:
        for line in format_experiment_result(result):
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

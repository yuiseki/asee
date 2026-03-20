"""Run split-aware owner embedding experiments on train/valid/test datasets."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np

from .compare_owner_embedding_strategies import (
    ReviewedSample,
    build_append_candidate_embeddings,
    build_classifier,
    collect_sample_embeddings,
    embeddings_from_samples,
    evaluate_review_samples,
)
from .enroll_owner import DEFAULT_OWNER_EMBED_PATH
from .overlay import GodModeOverlay
from .owner_policy import OWNER_COSINE_THRESHOLD, OWNER_TOPK
from .owner_rebuild_dataset import DEFAULT_OWNER_REBUILD_DATASET_ROOT
from .retrain_owner_embedding import (
    DEFAULT_SNAPSHOT_DIR,
    DatasetEvaluation,
    EmbeddingArray,
    OverlayLike,
    build_cosine_similarity_matrix,
    build_topk_values,
    greedy_select_false_negative_candidates,
    normalize_owner_embeddings,
)

DEFAULT_OWNER_REBUILD_SPLIT_DATASET_ROOT = DEFAULT_OWNER_REBUILD_DATASET_ROOT / "all-labeled-v1"


@dataclass(frozen=True, slots=True)
class SplitDataset:
    train_owner_positive: tuple[ReviewedSample, ...]
    train_guest_negative: tuple[ReviewedSample, ...]
    train_non_face_negative: tuple[ReviewedSample, ...]
    valid_owner_positive: tuple[ReviewedSample, ...]
    valid_guest_negative: tuple[ReviewedSample, ...]
    valid_non_face_negative: tuple[ReviewedSample, ...]
    test_owner_positive: tuple[ReviewedSample, ...]
    test_guest_negative: tuple[ReviewedSample, ...]
    test_non_face_negative: tuple[ReviewedSample, ...]

    @property
    def all_samples(self) -> tuple[ReviewedSample, ...]:
        return (
            *self.train_owner_positive,
            *self.train_guest_negative,
            *self.train_non_face_negative,
            *self.valid_owner_positive,
            *self.valid_guest_negative,
            *self.valid_non_face_negative,
            *self.test_owner_positive,
            *self.test_guest_negative,
            *self.test_non_face_negative,
        )

    @property
    def train_negative_all(self) -> tuple[ReviewedSample, ...]:
        return (*self.train_guest_negative, *self.train_non_face_negative)


@dataclass(frozen=True, slots=True)
class SplitExperimentEvaluation:
    train_owner_positive: DatasetEvaluation
    train_guest_negative: DatasetEvaluation
    train_non_face_negative: DatasetEvaluation
    valid_owner_positive: DatasetEvaluation
    valid_guest_negative: DatasetEvaluation
    valid_non_face_negative: DatasetEvaluation
    test_owner_positive: DatasetEvaluation
    test_guest_negative: DatasetEvaluation
    test_non_face_negative: DatasetEvaluation


@dataclass(frozen=True, slots=True)
class SplitExperimentResult:
    strategy_key: str
    bank_size: int
    selected_embeddings: int
    evaluation: SplitExperimentEvaluation
    candidate_embedding_path: Path | None = None
    selected_source_paths: tuple[Path, ...] = ()


@dataclass(frozen=True, slots=True)
class SplitExperimentReport:
    dataset_root: Path
    owner_embedding_path: Path
    current: SplitExperimentResult
    append_results: tuple[SplitExperimentResult, ...]
    rebuild_all: SplitExperimentResult
    rebuild_greedy_results: tuple[SplitExperimentResult, ...]
    prune_results: tuple[SplitExperimentResult, ...]
    summary_path: Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run split-aware append/rebuild experiments on labeled owner datasets"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_OWNER_REBUILD_SPLIT_DATASET_ROOT,
    )
    parser.add_argument(
        "--owner-embedding-path",
        type=Path,
        default=DEFAULT_OWNER_EMBED_PATH,
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=DEFAULT_SNAPSHOT_DIR,
    )
    parser.add_argument(
        "--negative-penalties",
        type=float,
        nargs="+",
        default=(3.0, 5.0),
    )
    parser.add_argument(
        "--prune-limits",
        type=int,
        nargs="+",
        default=(5, 10, 20, 40),
    )
    return parser


def load_split_dataset(dataset_root: Path) -> SplitDataset:
    manifest_path = dataset_root / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")

    bucket: dict[tuple[str, str], list[ReviewedSample]] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        split = str(row["split"])
        label = str(row["label"])
        if label not in {"owner_positive", "guest_negative", "non_face_negative"}:
            continue
        key = (split, label)
        bucket.setdefault(key, []).append(
            ReviewedSample(
                project_name=str(row["project_name"]),
                label=label,
                source_image=Path(str(row["materialized_image"])),
                source_sidecar=(
                    Path(str(row["materialized_sidecar"]))
                    if row.get("materialized_sidecar")
                    else None
                ),
                task_id=int(row["task_id"]),
            )
        )

    def _samples(split: str, label: str) -> tuple[ReviewedSample, ...]:
        return tuple(bucket.get((split, label), ()))

    return SplitDataset(
        train_owner_positive=_samples("train", "owner_positive"),
        train_guest_negative=_samples("train", "guest_negative"),
        train_non_face_negative=_samples("train", "non_face_negative"),
        valid_owner_positive=_samples("valid", "owner_positive"),
        valid_guest_negative=_samples("valid", "guest_negative"),
        valid_non_face_negative=_samples("valid", "non_face_negative"),
        test_owner_positive=_samples("test", "owner_positive"),
        test_guest_negative=_samples("test", "guest_negative"),
        test_non_face_negative=_samples("test", "non_face_negative"),
    )


def run_owner_rebuild_split_experiment(
    *,
    dataset_root: Path,
    owner_embedding_path: Path,
    snapshot_dir: Path,
    negative_penalties: tuple[float, ...] = (3.0, 5.0),
    prune_limits: tuple[int, ...] = (5, 10, 20, 40),
    overlay: OverlayLike | None = None,
    embedding_lookup: dict[Path, EmbeddingArray] | None = None,
) -> SplitExperimentReport:
    dataset = load_split_dataset(dataset_root)
    active_overlay = overlay or cast(OverlayLike, GodModeOverlay(width=1280, height=720))
    current_embeddings = normalize_owner_embeddings(
        cast(EmbeddingArray, np.load(owner_embedding_path))
    )
    sample_embeddings = collect_sample_embeddings(
        samples=dataset.all_samples,
        overlay=active_overlay,
        embedding_lookup=embedding_lookup,
    )
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = snapshot_dir / f"owner_rebuild_split_experiment_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    current = SplitExperimentResult(
        strategy_key="current",
        bank_size=int(current_embeddings.shape[0]),
        selected_embeddings=0,
        evaluation=_evaluate_split_dataset(
            dataset=dataset,
            owner_embeddings=current_embeddings,
            sample_embeddings=sample_embeddings,
            overlay=active_overlay,
        ),
    )

    append_results = tuple(
        _run_append_result(
            key=f"append_train_greedy_p{penalty:.1f}",
            current_embeddings=current_embeddings,
            dataset=dataset,
            sample_embeddings=sample_embeddings,
            overlay=active_overlay,
            negative_penalty=penalty,
            output_dir=output_dir,
        )
        for penalty in negative_penalties
    )

    rebuild_all_embeddings = _build_rebuild_all_embeddings(
        train_owner_positive=dataset.train_owner_positive,
        sample_embeddings=sample_embeddings,
    )
    rebuild_all_path = output_dir / "rebuild_train_all.npy"
    np.save(rebuild_all_path, rebuild_all_embeddings)
    rebuild_all = SplitExperimentResult(
        strategy_key="rebuild_train_all",
        bank_size=int(rebuild_all_embeddings.shape[0]),
        selected_embeddings=int(rebuild_all_embeddings.shape[0]),
        evaluation=_evaluate_split_dataset(
            dataset=dataset,
            owner_embeddings=rebuild_all_embeddings,
            sample_embeddings=sample_embeddings,
            overlay=active_overlay,
        ),
        candidate_embedding_path=rebuild_all_path,
        selected_source_paths=tuple(sample.source_image for sample in dataset.train_owner_positive),
    )

    rebuild_greedy_results = tuple(
        _run_rebuild_greedy_result(
            key=f"rebuild_train_greedy_p{penalty:.1f}",
            dataset=dataset,
            sample_embeddings=sample_embeddings,
            overlay=active_overlay,
            negative_penalty=penalty,
            output_dir=output_dir,
        )
        for penalty in negative_penalties
    )
    prune_results = tuple(
        _run_prune_result(
            key=f"prune_current_p{penalty:.1f}_n{limit}",
            current_embeddings=current_embeddings,
            dataset=dataset,
            sample_embeddings=sample_embeddings,
            overlay=active_overlay,
            negative_penalty=penalty,
            max_remove=limit,
            output_dir=output_dir,
        )
        for penalty in negative_penalties
        for limit in prune_limits
    )

    summary = {
        "dataset_root": str(dataset_root),
        "owner_embedding_path": str(owner_embedding_path),
        "current": _serialize_result(current),
        "append_results": [_serialize_result(result) for result in append_results],
        "rebuild_all": _serialize_result(rebuild_all),
        "rebuild_greedy_results": [
            _serialize_result(result) for result in rebuild_greedy_results
        ],
        "prune_results": [_serialize_result(result) for result in prune_results],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return SplitExperimentReport(
        dataset_root=dataset_root,
        owner_embedding_path=owner_embedding_path,
        current=current,
        append_results=append_results,
        rebuild_all=rebuild_all,
        rebuild_greedy_results=rebuild_greedy_results,
        prune_results=prune_results,
        summary_path=summary_path,
    )


def _run_append_result(
    *,
    key: str,
    current_embeddings: EmbeddingArray,
    dataset: SplitDataset,
    sample_embeddings: dict[Path, EmbeddingArray],
    overlay: OverlayLike,
    negative_penalty: float,
    output_dir: Path,
) -> SplitExperimentResult:
    owner_embeddings, selected_paths = build_append_candidate_embeddings(
        current_embeddings=current_embeddings,
        candidate_samples=dataset.train_owner_positive,
        negative_samples=dataset.train_negative_all,
        sample_embeddings=sample_embeddings,
        negative_penalty=negative_penalty,
        max_selected=None,
    )
    candidate_path = output_dir / f"{key}.npy"
    np.save(candidate_path, owner_embeddings)
    return SplitExperimentResult(
        strategy_key=key,
        bank_size=int(owner_embeddings.shape[0]),
        selected_embeddings=max(int(owner_embeddings.shape[0] - current_embeddings.shape[0]), 0),
        evaluation=_evaluate_split_dataset(
            dataset=dataset,
            owner_embeddings=owner_embeddings,
            sample_embeddings=sample_embeddings,
            overlay=overlay,
        ),
        candidate_embedding_path=candidate_path,
        selected_source_paths=selected_paths,
    )


def _run_rebuild_greedy_result(
    *,
    key: str,
    dataset: SplitDataset,
    sample_embeddings: dict[Path, EmbeddingArray],
    overlay: OverlayLike,
    negative_penalty: float,
    output_dir: Path,
) -> SplitExperimentResult:
    candidate_paths, candidate_embeddings = embeddings_from_samples(
        samples=dataset.train_owner_positive,
        sample_embeddings=sample_embeddings,
    )
    negative_paths, negative_embeddings = embeddings_from_samples(
        samples=dataset.train_negative_all,
        sample_embeddings=sample_embeddings,
    )
    if len(candidate_paths) == 0:
        raise RuntimeError("no train owner_positive embeddings available")

    positive_similarity = build_cosine_similarity_matrix(candidate_embeddings, candidate_embeddings)
    if len(negative_paths) == 0:
        negative_similarity = np.zeros((len(candidate_paths), 0), dtype=np.float32)
    else:
        negative_similarity = build_cosine_similarity_matrix(
            candidate_embeddings,
            negative_embeddings,
        )
    positive_topk_values = np.zeros((len(candidate_paths), OWNER_TOPK), dtype=np.float32)
    negative_topk_values = np.zeros((len(negative_paths), OWNER_TOPK), dtype=np.float32)
    selection = greedy_select_false_negative_candidates(
        candidate_paths=candidate_paths,
        positive_candidate_scores=positive_similarity,
        negative_candidate_scores=negative_similarity,
        positive_topk_values=positive_topk_values,
        negative_topk_values=negative_topk_values,
        threshold=OWNER_COSINE_THRESHOLD,
        negative_penalty=negative_penalty,
    )
    if selection.selected_indices:
        owner_embeddings = candidate_embeddings[list(selection.selected_indices)]
    else:
        owner_embeddings = candidate_embeddings[:1].copy()
    candidate_path = output_dir / f"{key}.npy"
    np.save(candidate_path, owner_embeddings)
    return SplitExperimentResult(
        strategy_key=key,
        bank_size=int(owner_embeddings.shape[0]),
        selected_embeddings=int(owner_embeddings.shape[0]),
        evaluation=_evaluate_split_dataset(
            dataset=dataset,
            owner_embeddings=owner_embeddings,
            sample_embeddings=sample_embeddings,
            overlay=overlay,
        ),
        candidate_embedding_path=candidate_path,
        selected_source_paths=selection.selected_paths,
    )


def _run_prune_result(
    *,
    key: str,
    current_embeddings: EmbeddingArray,
    dataset: SplitDataset,
    sample_embeddings: dict[Path, EmbeddingArray],
    overlay: OverlayLike,
    negative_penalty: float,
    max_remove: int,
    output_dir: Path,
) -> SplitExperimentResult:
    keep_indices = select_pruned_embedding_indices(
        current_embeddings=current_embeddings,
        positive_samples=dataset.train_owner_positive,
        negative_samples=dataset.train_negative_all,
        sample_embeddings=sample_embeddings,
        negative_penalty=negative_penalty,
        max_remove=max_remove,
    )
    owner_embeddings = current_embeddings[list(keep_indices)]
    candidate_path = output_dir / f"{key}.npy"
    np.save(candidate_path, owner_embeddings)
    return SplitExperimentResult(
        strategy_key=key,
        bank_size=int(owner_embeddings.shape[0]),
        selected_embeddings=int(current_embeddings.shape[0] - owner_embeddings.shape[0]),
        evaluation=_evaluate_split_dataset(
            dataset=dataset,
            owner_embeddings=owner_embeddings,
            sample_embeddings=sample_embeddings,
            overlay=overlay,
        ),
        candidate_embedding_path=candidate_path,
    )


def _build_rebuild_all_embeddings(
    *,
    train_owner_positive: tuple[ReviewedSample, ...],
    sample_embeddings: dict[Path, EmbeddingArray],
) -> EmbeddingArray:
    _, owner_embeddings = embeddings_from_samples(
        samples=train_owner_positive,
        sample_embeddings=sample_embeddings,
    )
    if owner_embeddings.shape[0] == 0:
        raise RuntimeError("no train owner_positive embeddings available")
    return owner_embeddings


def select_pruned_embedding_indices(
    *,
    current_embeddings: EmbeddingArray,
    positive_samples: tuple[ReviewedSample, ...],
    negative_samples: tuple[ReviewedSample, ...],
    sample_embeddings: dict[Path, EmbeddingArray],
    negative_penalty: float,
    max_remove: int,
) -> tuple[int, ...]:
    if max_remove <= 0:
        return tuple(range(int(current_embeddings.shape[0])))

    positive_paths, positive_embeddings = embeddings_from_samples(
        samples=positive_samples,
        sample_embeddings=sample_embeddings,
    )
    negative_paths, negative_embeddings = embeddings_from_samples(
        samples=negative_samples,
        sample_embeddings=sample_embeddings,
    )
    bank_size = int(current_embeddings.shape[0])
    if bank_size == 0:
        return ()

    positive_hits = np.zeros(bank_size, dtype=np.int32)
    negative_hits = np.zeros(bank_size, dtype=np.int32)

    if len(positive_paths) > 0:
        positive_similarity = build_cosine_similarity_matrix(
            current_embeddings,
            positive_embeddings,
        )
        positive_topk_values = build_topk_values(positive_similarity, topk=OWNER_TOPK)
        positive_owner_mask = positive_topk_values.mean(axis=1) >= OWNER_COSINE_THRESHOLD
        k = min(OWNER_TOPK, positive_similarity.shape[0])
        positive_top_indices = np.argpartition(
            positive_similarity,
            positive_similarity.shape[0] - k,
            axis=0,
        )[-k:]
        for column_index, is_owner in enumerate(positive_owner_mask):
            if not bool(is_owner):
                continue
            for embedding_index in positive_top_indices[:, column_index]:
                positive_hits[int(embedding_index)] += 1

    if len(negative_paths) > 0:
        negative_similarity = build_cosine_similarity_matrix(
            current_embeddings,
            negative_embeddings,
        )
        negative_topk_values = build_topk_values(negative_similarity, topk=OWNER_TOPK)
        negative_owner_mask = negative_topk_values.mean(axis=1) >= OWNER_COSINE_THRESHOLD
        k = min(OWNER_TOPK, negative_similarity.shape[0])
        negative_top_indices = np.argpartition(
            negative_similarity,
            negative_similarity.shape[0] - k,
            axis=0,
        )[-k:]
        for column_index, is_owner in enumerate(negative_owner_mask):
            if not bool(is_owner):
                continue
            for embedding_index in negative_top_indices[:, column_index]:
                negative_hits[int(embedding_index)] += 1

    score = negative_hits.astype(np.float32) * float(negative_penalty) - positive_hits.astype(
        np.float32
    )
    ranked = [index for index in np.argsort(score)[::-1] if score[index] > 0]
    to_remove = set(ranked[:max_remove])
    return tuple(index for index in range(bank_size) if index not in to_remove)


def _evaluate_split_dataset(
    *,
    dataset: SplitDataset,
    owner_embeddings: EmbeddingArray,
    sample_embeddings: dict[Path, EmbeddingArray],
    overlay: OverlayLike,
) -> SplitExperimentEvaluation:
    classifier = build_classifier(overlay=overlay, owner_embeddings=owner_embeddings)
    return SplitExperimentEvaluation(
        train_owner_positive=evaluate_review_samples(
            samples=dataset.train_owner_positive,
            sample_embeddings=sample_embeddings,
            classify_embedding=classifier,
        ),
        train_guest_negative=evaluate_review_samples(
            samples=dataset.train_guest_negative,
            sample_embeddings=sample_embeddings,
            classify_embedding=classifier,
        ),
        train_non_face_negative=evaluate_review_samples(
            samples=dataset.train_non_face_negative,
            sample_embeddings=sample_embeddings,
            classify_embedding=classifier,
        ),
        valid_owner_positive=evaluate_review_samples(
            samples=dataset.valid_owner_positive,
            sample_embeddings=sample_embeddings,
            classify_embedding=classifier,
        ),
        valid_guest_negative=evaluate_review_samples(
            samples=dataset.valid_guest_negative,
            sample_embeddings=sample_embeddings,
            classify_embedding=classifier,
        ),
        valid_non_face_negative=evaluate_review_samples(
            samples=dataset.valid_non_face_negative,
            sample_embeddings=sample_embeddings,
            classify_embedding=classifier,
        ),
        test_owner_positive=evaluate_review_samples(
            samples=dataset.test_owner_positive,
            sample_embeddings=sample_embeddings,
            classify_embedding=classifier,
        ),
        test_guest_negative=evaluate_review_samples(
            samples=dataset.test_guest_negative,
            sample_embeddings=sample_embeddings,
            classify_embedding=classifier,
        ),
        test_non_face_negative=evaluate_review_samples(
            samples=dataset.test_non_face_negative,
            sample_embeddings=sample_embeddings,
            classify_embedding=classifier,
        ),
    )


def _serialize_dataset_evaluation(evaluation: DatasetEvaluation) -> dict[str, int | float]:
    return asdict(evaluation)


def _serialize_result(result: SplitExperimentResult) -> dict[str, object]:
    return {
        "strategy_key": result.strategy_key,
        "bank_size": result.bank_size,
        "selected_embeddings": result.selected_embeddings,
        "candidate_embedding_path": (
            str(result.candidate_embedding_path) if result.candidate_embedding_path else None
        ),
        "selected_source_paths": [str(path) for path in result.selected_source_paths],
        "evaluation": {
            "train_owner_positive": _serialize_dataset_evaluation(
                result.evaluation.train_owner_positive
            ),
            "train_guest_negative": _serialize_dataset_evaluation(
                result.evaluation.train_guest_negative
            ),
            "train_non_face_negative": _serialize_dataset_evaluation(
                result.evaluation.train_non_face_negative
            ),
            "valid_owner_positive": _serialize_dataset_evaluation(
                result.evaluation.valid_owner_positive
            ),
            "valid_guest_negative": _serialize_dataset_evaluation(
                result.evaluation.valid_guest_negative
            ),
            "valid_non_face_negative": _serialize_dataset_evaluation(
                result.evaluation.valid_non_face_negative
            ),
            "test_owner_positive": _serialize_dataset_evaluation(
                result.evaluation.test_owner_positive
            ),
            "test_guest_negative": _serialize_dataset_evaluation(
                result.evaluation.test_guest_negative
            ),
            "test_non_face_negative": _serialize_dataset_evaluation(
                result.evaluation.test_non_face_negative
            ),
        },
    }


def main() -> int:
    args = build_arg_parser().parse_args()
    report = run_owner_rebuild_split_experiment(
        dataset_root=Path(args.dataset_root),
        owner_embedding_path=Path(args.owner_embedding_path),
        snapshot_dir=Path(args.snapshot_dir),
        negative_penalties=tuple(float(value) for value in args.negative_penalties),
        prune_limits=tuple(int(value) for value in args.prune_limits),
    )
    print(f"summary={report.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

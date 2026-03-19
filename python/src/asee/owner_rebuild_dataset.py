"""Prepare a reproducible train/valid/test dataset from labeled face-review projects."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .compare_owner_embedding_strategies import (
    DEFAULT_BASELINE_CONTACTS_BACKUP_ROOT,
    DEFAULT_BASELINE_HOLDOUT_BACKUP_ROOT,
    DEFAULT_BASELINE_MAKEUP_BACKUP_ROOT,
    DEFAULT_DARK_ROOM_MORNING_BACKUP_ROOT,
    DEFAULT_HARD_POSITIVE_BACKUP_ROOT,
    DEFAULT_NON_FACE_HARD_NEGATIVE_BACKUP_ROOT,
    ReviewedSample,
    load_review_samples,
    resolve_latest_export_json,
)

DEFAULT_GUEST_FIRST_WAVE_BACKUP_ROOT = Path(
    "/home/yuiseki/Workspaces/private/datasets/faces/golden_review_backups/guest-session-first-wave-2026-03-19-v1"
)
DEFAULT_GUEST_SECOND_WAVE_BACKUP_ROOT = Path(
    "/home/yuiseki/Workspaces/private/datasets/faces/golden_review_backups/guest-session-second-wave-2026-03-19-v1"
)
DEFAULT_OWNER_REBUILD_DATASET_ROOT = Path(
    "/home/yuiseki/Workspaces/private/datasets/faces/owner_rebuild_dataset"
)

SplitName = Literal["train", "valid", "test"]


@dataclass(frozen=True, slots=True)
class SplitRatios:
    train: float = 0.7
    valid: float = 0.15
    test: float = 0.15


@dataclass(frozen=True, slots=True)
class LabeledDatasetSummary:
    dataset_root: Path
    manifest_path: Path
    split_counts: dict[str, dict[str, int]]
    project_exports: dict[str, str]
    seed: int


def default_project_exports() -> dict[str, Path]:
    return {
        "owner_hard_positive_glasses": resolve_latest_export_json(
            DEFAULT_HARD_POSITIVE_BACKUP_ROOT
        ),
        "owner_baseline_contacts": resolve_latest_export_json(
            DEFAULT_BASELINE_CONTACTS_BACKUP_ROOT
        ),
        "owner_baseline_makeup": resolve_latest_export_json(
            DEFAULT_BASELINE_MAKEUP_BACKUP_ROOT
        ),
        "owner_non_face_hard_negatives": resolve_latest_export_json(
            DEFAULT_NON_FACE_HARD_NEGATIVE_BACKUP_ROOT
        ),
        "owner_baseline_holdout": resolve_latest_export_json(
            DEFAULT_BASELINE_HOLDOUT_BACKUP_ROOT
        ),
        "owner_dark_room_morning": resolve_latest_export_json(
            DEFAULT_DARK_ROOM_MORNING_BACKUP_ROOT
        ),
        "guest_session_first_wave": resolve_latest_export_json(
            DEFAULT_GUEST_FIRST_WAVE_BACKUP_ROOT
        ),
        "guest_session_second_wave": resolve_latest_export_json(
            DEFAULT_GUEST_SECOND_WAVE_BACKUP_ROOT
        ),
    }


def load_all_labeled_review_samples(
    project_exports: dict[str, Path],
) -> tuple[ReviewedSample, ...]:
    samples: list[ReviewedSample] = []
    for project_name, export_path in project_exports.items():
        samples.extend(load_review_samples(export_path, project_name=project_name))
    return tuple(samples)


def _hashed_order_key(sample: ReviewedSample, *, seed: int) -> str:
    payload = (
        f"{seed}|{sample.project_name}|{sample.label}|{sample.task_id}|{sample.source_image}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _compute_split_sizes(total: int, ratios: SplitRatios) -> dict[SplitName, int]:
    if total <= 0:
        return {"train": 0, "valid": 0, "test": 0}

    if total == 1:
        return {"train": 1, "valid": 0, "test": 0}
    if total == 2:
        return {"train": 1, "valid": 0, "test": 1}

    train = int(total * ratios.train)
    valid = int(total * ratios.valid)
    test = total - train - valid

    if valid == 0:
        valid = 1
        train = max(1, train - 1)
    if test == 0:
        test = 1
        if train > valid:
            train = max(1, train - 1)
        else:
            valid = max(1, valid - 1)

    while train + valid + test < total:
        train += 1
    while train + valid + test > total:
        if train >= valid and train >= test and train > 1:
            train -= 1
        elif valid >= test and valid > 1:
            valid -= 1
        elif test > 1:
            test -= 1
        else:
            break

    return {"train": train, "valid": valid, "test": test}


def split_review_samples(
    samples: tuple[ReviewedSample, ...],
    *,
    seed: int = 20260320,
    ratios: SplitRatios | None = None,
) -> dict[SplitName, tuple[ReviewedSample, ...]]:
    active_ratios = ratios if ratios is not None else SplitRatios()
    by_label: dict[str, list[ReviewedSample]] = defaultdict(list)
    for sample in samples:
        by_label[sample.label].append(sample)

    split_to_samples: dict[SplitName, list[ReviewedSample]] = {
        "train": [],
        "valid": [],
        "test": [],
    }

    for _label, label_samples in sorted(by_label.items()):
        ordered = sorted(label_samples, key=lambda sample: _hashed_order_key(sample, seed=seed))
        sizes = _compute_split_sizes(len(ordered), active_ratios)
        start = 0
        for split_name in ("train", "valid", "test"):
            count = sizes[split_name]
            split_to_samples[split_name].extend(ordered[start : start + count])
            start += count

    return {name: tuple(values) for name, values in split_to_samples.items()}


def _safe_name(text: str) -> str:
    return "".join(
        char if char.isalnum() or char in {"-", "_"} else "-" for char in text
    ).strip("-")


def _copy_or_symlink(src: Path, dst: Path, *, copy_files: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_files:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src)


def materialize_split_dataset(
    *,
    split_samples: dict[SplitName, tuple[ReviewedSample, ...]],
    output_root: Path,
    project_exports: dict[str, Path],
    seed: int,
    copy_files: bool = False,
) -> LabeledDatasetSummary:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    split_counts: dict[str, dict[str, int]] = {
        split_name: {}
        for split_name in ("train", "valid", "test")
    }

    for split_name, samples in split_samples.items():
        for index, sample in enumerate(samples, start=1):
            project_key = _safe_name(sample.project_name)
            sample_dir = output_root / split_name / sample.label / project_key
            filename = f"{index:04d}-{sample.source_image.name}"
            image_target = sample_dir / filename
            _copy_or_symlink(sample.source_image, image_target, copy_files=copy_files)
            sidecar_target: Path | None = None
            if sample.source_sidecar is not None and sample.source_sidecar.exists():
                sidecar_target = image_target.with_suffix(".json")
                _copy_or_symlink(sample.source_sidecar, sidecar_target, copy_files=copy_files)

            split_counts[split_name][sample.label] = (
                split_counts[split_name].get(sample.label, 0) + 1
            )
            manifest_rows.append(
                {
                    "split": split_name,
                    "label": sample.label,
                    "project_name": sample.project_name,
                    "task_id": sample.task_id,
                    "source_image": str(sample.source_image),
                    "source_sidecar": (
                        str(sample.source_sidecar) if sample.source_sidecar is not None else None
                    ),
                    "materialized_image": str(image_target),
                    "materialized_sidecar": (
                        str(sidecar_target) if sidecar_target is not None else None
                    ),
                }
            )

    manifest_path = output_root / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in manifest_rows) + "\n",
        encoding="utf-8",
    )
    summary = {
        "dataset_root": str(output_root),
        "seed": seed,
        "project_exports": {name: str(path) for name, path in project_exports.items()},
        "split_counts": {
            split_name: dict(counts) for split_name, counts in split_counts.items()
        },
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return LabeledDatasetSummary(
        dataset_root=output_root,
        manifest_path=manifest_path,
        split_counts={split_name: dict(counts) for split_name, counts in split_counts.items()},
        project_exports={name: str(path) for name, path in project_exports.items()},
        seed=seed,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a train/valid/test dataset from labeled ASEE face review projects"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OWNER_REBUILD_DATASET_ROOT / "all-labeled-v1",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260320,
    )
    parser.add_argument(
        "--copy-files",
        action="store_true",
        help="Copy files instead of creating symlinks.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    project_exports = default_project_exports()
    samples = load_all_labeled_review_samples(project_exports)
    split_samples = split_review_samples(samples, seed=int(args.seed))
    summary = materialize_split_dataset(
        split_samples=split_samples,
        output_root=Path(args.output_root),
        project_exports=project_exports,
        seed=int(args.seed),
        copy_files=bool(args.copy_files),
    )
    print(f"dataset_root={summary.dataset_root}")
    print(f"manifest={summary.manifest_path}")
    print(json.dumps(summary.split_counts, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

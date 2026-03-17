from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast


@dataclass(frozen=True)
class ReviewSample:
    label: str
    source_root: Path
    image_path: Path
    sidecar_path: Path
    day: str
    hour_bucket: str
    camera_id: int | None
    captured_at: str
    score: float | None
    room_context_present: bool


def _parse_sidecar(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def load_review_samples(*, source_root: Path, label: str) -> list[ReviewSample]:
    samples: list[ReviewSample] = []
    for sidecar_path in sorted(source_root.rglob("*.json")):
        image_path = sidecar_path.with_suffix(".jpg")
        if not image_path.exists():
            continue
        data = _parse_sidecar(sidecar_path)
        rel_parts = sidecar_path.relative_to(source_root).parts
        if len(rel_parts) < 5:
            continue
        day = "-".join(rel_parts[:3])
        hour = rel_parts[3]
        hour_bucket = "05" if hour == "05" else "06"
        camera_id = data.get("cameraId")
        if not isinstance(camera_id, int):
            camera_id = None
        score = data.get("score")
        if not isinstance(score, (int, float)):
            score = None
        captured_at = str(data.get("capturedAt") or image_path.stem)
        room_context_present = isinstance(data.get("roomContext"), dict)
        samples.append(
            ReviewSample(
                label=label,
                source_root=source_root,
                image_path=image_path,
                sidecar_path=sidecar_path,
                day=day,
                hour_bucket=hour_bucket,
                camera_id=camera_id,
                captured_at=captured_at,
                score=float(score) if score is not None else None,
                room_context_present=room_context_present,
            )
        )
    return samples


def _stratum_key(sample: ReviewSample) -> tuple[str, int, str]:
    return (
        sample.day,
        -1 if sample.camera_id is None else int(sample.camera_id),
        sample.hour_bucket,
    )


def select_representative_samples(
    samples: list[ReviewSample],
    *,
    target_count: int,
) -> list[ReviewSample]:
    if target_count <= 0 or not samples:
        return []

    buckets: dict[tuple[str, int, str], list[ReviewSample]] = defaultdict(list)
    for sample in sorted(samples, key=lambda item: (item.captured_at, str(item.image_path))):
        buckets[_stratum_key(sample)].append(sample)

    order = sorted(
        buckets,
        key=lambda key: (
            -len(buckets[key]),
            key[0],
            key[1],
            key[2],
        ),
    )

    selected: list[ReviewSample] = []
    indexes = {key: 0 for key in order}
    while len(selected) < target_count:
        progressed = False
        for key in order:
            idx = indexes[key]
            group = buckets[key]
            if idx >= len(group):
                continue
            selected.append(group[idx])
            indexes[key] = idx + 1
            progressed = True
            if len(selected) >= target_count:
                break
        if not progressed:
            break
    return selected


def materialize_subset(*, output_root: Path, samples: list[ReviewSample]) -> dict[str, Any]:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.jsonl"
    summary_path = output_root / "summary.json"

    counts: dict[str, int] = defaultdict(int)
    by_day: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_camera: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    room_context_counts: dict[str, int] = defaultdict(int)

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for sample in samples:
            rel = sample.image_path.relative_to(sample.source_root)
            dest_image = output_root / sample.label / rel
            dest_sidecar = dest_image.with_suffix(".json")
            dest_image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sample.image_path, dest_image)
            shutil.copy2(sample.sidecar_path, dest_sidecar)
            counts[sample.label] += 1
            by_day[sample.label][sample.day] += 1
            by_camera[sample.label][str(sample.camera_id)] += 1
            if sample.room_context_present:
                room_context_counts[sample.label] += 1
            manifest.write(
                json.dumps(
                    {
                        **asdict(sample),
                        "source_root": str(sample.source_root),
                        "image_path": str(sample.image_path),
                        "sidecar_path": str(sample.sidecar_path),
                        "copied_image": str(dest_image),
                        "copied_sidecar": str(dest_sidecar),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = {
        "sampleCount": len(samples),
        "counts": dict(counts),
        "byDay": {label: dict(sorted(values.items())) for label, values in by_day.items()},
        "byCamera": {label: dict(sorted(values.items())) for label, values in by_camera.items()},
        "roomContextPresent": dict(room_context_counts),
        "manifestPath": str(manifest_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Curate a representative review subset")
    parser.add_argument("--owner-root", type=Path, required=True)
    parser.add_argument("--subject-root", type=Path, required=True)
    parser.add_argument("--owner-target", type=int, default=80)
    parser.add_argument("--subject-target", type=int, default=160)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    owner_samples = load_review_samples(source_root=args.owner_root, label="owner_raw")
    subject_samples = load_review_samples(
        source_root=args.subject_root,
        label="subject_false_negative",
    )
    selected = [
        *select_representative_samples(owner_samples, target_count=int(args.owner_target)),
        *select_representative_samples(subject_samples, target_count=int(args.subject_target)),
    ]
    summary = materialize_subset(output_root=args.output_root, samples=selected)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

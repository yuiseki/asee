"""Collect owner-only camera disagreement false negatives into a dedicated bucket."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from shutil import copy2
from typing import Any

from .retrain_owner_embedding import iter_image_paths

DEFAULT_SUBJECT_SOURCE_DIR = Path(
    "/home/yuiseki/Workspaces/private/datasets/faces/others/2026/03/14"
)
DEFAULT_OWNER_SOURCE_DIR = Path(
    "/home/yuiseki/Workspaces/private/datasets/faces/_raw/2026/03/14"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/yuiseki/Workspaces/private/datasets/faces/owner_camera_disagreement_session"
)
DEFAULT_WINDOW_SECONDS = 2.0


@dataclass(frozen=True, slots=True)
class CaptureEvent:
    source_path: Path
    label: str
    captured_at: datetime
    camera_id: int | None
    score: float
    owner_count: int | None
    subject_count: int | None
    people_count: int | None


@dataclass(frozen=True, slots=True)
class OwnerCameraDisagreementFeature:
    subject_event: CaptureEvent
    matched_owner_event: CaptureEvent
    matched_delta_seconds: float


@dataclass(frozen=True, slots=True)
class OwnerCameraDisagreementResult:
    output_root: Path
    manifest_path: Path
    total_selected: int


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect SUBJECT false negatives whose nearby sibling camera captured the same "
            "owner as OWNER."
        )
    )
    parser.add_argument("--subject-source", type=Path, default=DEFAULT_SUBJECT_SOURCE_DIR)
    parser.add_argument("--owner-source", type=Path, default=DEFAULT_OWNER_SOURCE_DIR)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    return parser


def load_capture_events(root: Path) -> tuple[CaptureEvent, ...]:
    events: list[CaptureEvent] = []
    for image_path in iter_image_paths(root):
        sidecar_path = image_path.with_suffix(".json")
        payload = _load_sidecar_payload(sidecar_path)
        captured_at = _parse_captured_at(payload.get("capturedAt"))
        if captured_at is None:
            continue
        raw_frame_counts = payload.get("frameCounts")
        frame_counts = raw_frame_counts if isinstance(raw_frame_counts, dict) else {}
        events.append(
            CaptureEvent(
                source_path=image_path,
                label=str(payload.get("label", "")),
                captured_at=captured_at,
                camera_id=_safe_int(payload.get("cameraId")),
                score=_safe_float(payload.get("score")),
                owner_count=_safe_int(frame_counts.get("ownerCount")),
                subject_count=_safe_int(frame_counts.get("subjectCount")),
                people_count=_safe_int(frame_counts.get("peopleCount")),
            )
        )
    return tuple(sorted(events, key=lambda event: (event.captured_at, str(event.source_path))))


def select_owner_camera_disagreement_features(
    *,
    subject_events: Sequence[CaptureEvent],
    owner_events: Sequence[CaptureEvent],
    window_seconds: float,
) -> tuple[OwnerCameraDisagreementFeature, ...]:
    selected: list[OwnerCameraDisagreementFeature] = []
    for subject_event in subject_events:
        if not _is_single_subject_event(subject_event):
            continue
        best_match: CaptureEvent | None = None
        best_delta: float | None = None
        for owner_event in owner_events:
            if not _is_single_owner_event(owner_event):
                continue
            if (
                subject_event.camera_id is not None
                and owner_event.camera_id is not None
                and subject_event.camera_id == owner_event.camera_id
            ):
                continue
            delta_seconds = abs(
                (owner_event.captured_at - subject_event.captured_at).total_seconds()
            )
            if delta_seconds > window_seconds:
                continue
            better_same_delta = (
                delta_seconds == best_delta
                and owner_event.score > (best_match.score if best_match else -1.0)
            )
            if best_delta is None or delta_seconds < best_delta or better_same_delta:
                best_match = owner_event
                best_delta = delta_seconds
        if best_match is None or best_delta is None:
            continue
        selected.append(
            OwnerCameraDisagreementFeature(
                subject_event=subject_event,
                matched_owner_event=best_match,
                matched_delta_seconds=float(best_delta),
            )
        )
    return tuple(selected)


def copy_owner_camera_disagreement_features(
    *,
    subject_root: Path,
    selected_features: Sequence[OwnerCameraDisagreementFeature],
    output_root: Path,
    copy_file: Callable[[Path, Path], object] | None = None,
) -> OwnerCameraDisagreementResult:
    writer = copy_file or copy2
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for feature in selected_features:
            subject_path = feature.subject_event.source_path
            relative_path = subject_path.relative_to(subject_root)
            destination = output_root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            writer(subject_path, destination)
            sidecar_path = subject_path.with_suffix(".json")
            if sidecar_path.exists():
                writer(sidecar_path, destination.with_suffix(".json"))
            manifest.write(
                json.dumps(
                    {
                        "source_path": str(subject_path),
                        "relative_path": str(relative_path),
                        "subject_camera_id": feature.subject_event.camera_id,
                        "subject_score": feature.subject_event.score,
                        "matched_owner_source_path": str(feature.matched_owner_event.source_path),
                        "matched_owner_camera_id": feature.matched_owner_event.camera_id,
                        "matched_owner_score": feature.matched_owner_event.score,
                        "matched_delta_seconds": feature.matched_delta_seconds,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            manifest.write("\n")
    return OwnerCameraDisagreementResult(
        output_root=output_root,
        manifest_path=manifest_path,
        total_selected=len(selected_features),
    )


def default_output_root_for_source(source: Path) -> Path:
    if len(source.parts) >= 3 and all(part.isdigit() for part in source.parts[-3:]):
        return DEFAULT_OUTPUT_ROOT / "-".join(source.parts[-3:])
    return DEFAULT_OUTPUT_ROOT / source.name


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    subject_source = Path(args.subject_source)
    owner_source = Path(args.owner_source)
    output_root = (
        Path(args.output_root)
        if args.output_root is not None
        else default_output_root_for_source(subject_source)
    )
    selected = select_owner_camera_disagreement_features(
        subject_events=load_capture_events(subject_source),
        owner_events=load_capture_events(owner_source),
        window_seconds=float(args.window_seconds),
    )
    result = copy_owner_camera_disagreement_features(
        subject_root=subject_source,
        selected_features=selected,
        output_root=output_root,
    )
    print(f"output_root={result.output_root}")
    print(f"manifest={result.manifest_path}")
    print(f"total_selected={result.total_selected}")
    return 0


def _load_sidecar_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _parse_captured_at(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _is_single_subject_event(event: CaptureEvent) -> bool:
    return (
        event.label == "SUBJECT"
        and event.subject_count == 1
        and event.people_count == 1
        and (event.owner_count is None or event.owner_count == 0)
    )


def _is_single_owner_event(event: CaptureEvent) -> bool:
    return (
        event.label == "OWNER"
        and event.owner_count == 1
        and event.people_count == 1
        and (event.subject_count is None or event.subject_count == 0)
    )


if __name__ == "__main__":
    raise SystemExit(main())

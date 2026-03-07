"""Helpers for choosing the GOD MODE multi-camera layout."""

from __future__ import annotations

import argparse
import re
import subprocess
from collections.abc import Iterable

VIDEO_RE = re.compile(r"/dev/video(\d+)")


def parse_v4l2_devices(output: str) -> list[tuple[str, int]]:
    """Parse `v4l2-ctl --list-devices` output into (label, primary_video_index)."""
    devices: list[tuple[str, int]] = []
    label = ""
    first_video_index: int | None = None

    for raw_line in str(output or "").splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            if label and first_video_index is not None:
                devices.append((label, first_video_index))
            label = ""
            first_video_index = None
            continue

        if not raw_line.startswith((" ", "\t")):
            if label and first_video_index is not None:
                devices.append((label, first_video_index))
            label = line.rstrip(":")
            first_video_index = None
            continue

        if first_video_index is None:
            match = VIDEO_RE.search(line)
            if match:
                first_video_index = int(match.group(1))

    if label and first_video_index is not None:
        devices.append((label, first_video_index))

    return devices


def extend_with_optional_camera(
    base_cameras: Iterable[int],
    devices: Iterable[tuple[str, int]],
    preferred_tokens: Iterable[str] = ("anker",),
) -> list[int]:
    """Append one optional extra camera, preferring labels that match tokens."""
    base: list[int] = []
    seen: set[int] = set()
    for camera in base_cameras:
        camera_index = int(camera)
        if camera_index in seen:
            continue
        seen.add(camera_index)
        base.append(camera_index)

    extras = [(label, index) for label, index in devices if index not in seen]
    if not extras:
        return base

    lowered_tokens = tuple(token.lower() for token in preferred_tokens if str(token).strip())
    preferred: int | None = None
    if lowered_tokens:
        for label, index in extras:
            lowered_label = label.lower()
            if any(token in lowered_label for token in lowered_tokens):
                preferred = index
                break

    base.append(preferred if preferred is not None else extras[0][1])
    return base


def detect_v4l2_devices() -> list[tuple[str, int]]:
    """Probe connected V4L2 devices via `v4l2-ctl --list-devices`."""
    try:
        proc = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return parse_v4l2_devices(proc.stdout)


def parse_camera_csv(raw: str) -> list[int]:
    """Parse camera CSV text into numeric device indices."""
    return [int(part.strip()) for part in str(raw or "").split(",") if part.strip()]


def build_camera_csv(base: str, preferred_tokens: Iterable[str]) -> str:
    """Build camera CSV by appending at most one preferred optional device."""
    cameras = extend_with_optional_camera(
        parse_camera_csv(base),
        detect_v4l2_devices(),
        preferred_tokens=preferred_tokens,
    )
    return ",".join(str(camera) for camera in cameras)


def main() -> int:
    """CLI entrypoint for computing GOD MODE camera CSV."""
    parser = argparse.ArgumentParser(
        description="Build GOD MODE camera CSV with one optional extra camera.",
    )
    parser.add_argument("--base", default="0,2,4", help="Base camera CSV")
    parser.add_argument(
        "--prefer",
        default="anker",
        help="Comma-separated label tokens to prefer for the optional camera",
    )
    args = parser.parse_args()

    preferred_tokens = [part.strip() for part in args.prefer.split(",") if part.strip()]
    print(build_camera_csv(args.base, preferred_tokens))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

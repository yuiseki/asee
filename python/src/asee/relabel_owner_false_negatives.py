"""Move current guest-bucket crops into an owner false-negative bucket."""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Iterable, Sequence
from pathlib import Path

DEFAULT_OTHERS_DIR = Path("/home/yuiseki/Workspaces/private/datasets/faces/others")
DEFAULT_OWNER_FALSE_NEGATIVE_DIR = Path(
    "/home/yuiseki/Workspaces/private/datasets/faces/owner_false_negative"
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for false-negative relabeling."""
    parser = argparse.ArgumentParser(
        description=(
            "Move current face crops from the guest bucket into the "
            "owner false-negative bucket."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_OTHERS_DIR,
        help=f"Current guest bucket directory (default: {DEFAULT_OTHERS_DIR})",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_OWNER_FALSE_NEGATIVE_DIR,
        help=(
            "Destination directory for owner false-negative crops "
            f"(default: {DEFAULT_OWNER_FALSE_NEGATIVE_DIR})"
        ),
    )
    return parser


def iter_source_files(source: Path) -> Iterable[Path]:
    """Yield files under the source tree in stable order."""
    if not source.exists():
        return ()
    return (path for path in sorted(source.rglob("*")) if path.is_file())


def relabel_owner_false_negatives(*, source: Path, destination: Path) -> int:
    """Move current guest crops into the owner false-negative bucket."""
    moved = 0
    for path in list(iter_source_files(source)):
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))
        moved += 1
    _remove_empty_dirs(source)
    return moved


def _remove_empty_dirs(root: Path) -> None:
    """Remove empty directories left behind after moves."""
    if not root.exists():
        return
    for directory in sorted((path for path in root.rglob("*") if path.is_dir()), reverse=True):
        if not any(directory.iterdir()):
            directory.rmdir()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for relabeling current owner false negatives."""
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    moved = relabel_owner_false_negatives(
        source=Path(args.source),
        destination=Path(args.destination),
    )
    print(f"moved {moved} face crops")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

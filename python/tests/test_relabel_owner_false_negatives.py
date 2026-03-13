from __future__ import annotations

from pathlib import Path

import pytest

from asee.relabel_owner_false_negatives import (
    DEFAULT_OTHERS_DIR,
    DEFAULT_OWNER_FALSE_NEGATIVE_DIR,
    build_arg_parser,
    main,
    relabel_owner_false_negatives,
)


def test_build_arg_parser_uses_dataset_defaults() -> None:
    parser = build_arg_parser()

    args = parser.parse_args([])

    assert args.source == DEFAULT_OTHERS_DIR
    assert args.destination == DEFAULT_OWNER_FALSE_NEGATIVE_DIR


def test_relabel_owner_false_negatives_moves_files_preserving_structure(
    tmp_path: Path,
) -> None:
    source = tmp_path / "others"
    destination = tmp_path / "owner_false_negative"
    first = source / "2026" / "03" / "13" / "18" / "first.jpg"
    second = source / "2026" / "03" / "13" / "19" / "second.jpg"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    moved = relabel_owner_false_negatives(source=source, destination=destination)

    assert moved == 2
    assert not first.exists()
    assert not second.exists()
    assert (destination / "2026" / "03" / "13" / "18" / "first.jpg").read_bytes() == b"first"
    assert (destination / "2026" / "03" / "13" / "19" / "second.jpg").read_bytes() == b"second"


def test_relabel_owner_false_negatives_is_noop_for_empty_source(tmp_path: Path) -> None:
    source = tmp_path / "others"
    destination = tmp_path / "owner_false_negative"
    source.mkdir(parents=True)

    moved = relabel_owner_false_negatives(source=source, destination=destination)

    assert moved == 0
    assert not destination.exists()


def test_main_prints_moved_count(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "others"
    destination = tmp_path / "owner_false_negative"
    sample = source / "2026" / "03" / "13" / "18" / "sample.jpg"
    sample.parent.mkdir(parents=True)
    sample.write_bytes(b"sample")

    exit_code = main(["--source", str(source), "--destination", str(destination)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == "moved 1 face crops"

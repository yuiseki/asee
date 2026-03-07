"""Tests for resolving model and embedding assets across asee and legacy GOD MODE."""

from __future__ import annotations

from pathlib import Path

from asee.model_assets import candidate_model_asset_paths, resolve_model_asset_path


def test_candidate_model_asset_paths_prefers_package_then_legacy(tmp_path: Path) -> None:
    package_dir = tmp_path / "package-models"
    legacy_dir = tmp_path / "legacy-models"
    package_dir.mkdir()
    legacy_dir.mkdir()

    paths = candidate_model_asset_paths(
        "owner_embedding.npy",
        package_dir=package_dir,
        legacy_dir=legacy_dir,
    )

    assert paths == (
        package_dir / "owner_embedding.npy",
        legacy_dir / "owner_embedding.npy",
    )


def test_resolve_model_asset_path_prefers_package_copy(tmp_path: Path) -> None:
    package_dir = tmp_path / "package-models"
    legacy_dir = tmp_path / "legacy-models"
    package_dir.mkdir()
    legacy_dir.mkdir()
    package_path = package_dir / "face_detection_yunet_2023mar.onnx"
    legacy_path = legacy_dir / "face_detection_yunet_2023mar.onnx"
    package_path.write_bytes(b"package")
    legacy_path.write_bytes(b"legacy")

    resolved = resolve_model_asset_path(
        "face_detection_yunet_2023mar.onnx",
        package_dir=package_dir,
        legacy_dir=legacy_dir,
    )

    assert resolved == package_path


def test_resolve_model_asset_path_falls_back_to_legacy_copy(tmp_path: Path) -> None:
    package_dir = tmp_path / "package-models"
    legacy_dir = tmp_path / "legacy-models"
    package_dir.mkdir()
    legacy_dir.mkdir()
    legacy_path = legacy_dir / "face_recognition_sface_2021dec.onnx"
    legacy_path.write_bytes(b"legacy")

    resolved = resolve_model_asset_path(
        "face_recognition_sface_2021dec.onnx",
        package_dir=package_dir,
        legacy_dir=legacy_dir,
    )

    assert resolved == legacy_path


def test_resolve_model_asset_path_returns_package_destination_when_missing(tmp_path: Path) -> None:
    package_dir = tmp_path / "package-models"
    legacy_dir = tmp_path / "legacy-models"
    package_dir.mkdir()
    legacy_dir.mkdir()

    resolved = resolve_model_asset_path(
        "owner_embedding.npy",
        package_dir=package_dir,
        legacy_dir=legacy_dir,
    )

    assert resolved == package_dir / "owner_embedding.npy"

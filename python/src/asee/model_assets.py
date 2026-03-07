"""Resolve model and embedding assets across asee and legacy tmp/GOD_MODE."""

from __future__ import annotations

from pathlib import Path

PACKAGE_MODELS_DIR = Path(__file__).resolve().parent / "models"


def discover_legacy_models_dir(start: Path | None = None) -> Path | None:
    """Discover the legacy tmp/GOD_MODE models directory from the current repo layout."""
    anchor = start or Path(__file__).resolve()
    for parent in anchor.parents:
        candidate = parent / "tmp" / "GOD_MODE" / "models"
        if candidate.exists():
            return candidate
    return None


LEGACY_MODELS_DIR = discover_legacy_models_dir()


def candidate_model_asset_paths(
    filename: str,
    *,
    package_dir: Path = PACKAGE_MODELS_DIR,
    legacy_dir: Path | None = LEGACY_MODELS_DIR,
) -> tuple[Path, ...]:
    """Return package-first candidate paths for the requested asset."""
    candidates = [package_dir / filename]
    if legacy_dir is not None:
        candidates.append(legacy_dir / filename)
    return tuple(candidates)


def resolve_model_asset_path(
    filename: str,
    *,
    package_dir: Path = PACKAGE_MODELS_DIR,
    legacy_dir: Path | None = LEGACY_MODELS_DIR,
) -> Path:
    """Resolve an asset path, falling back to tmp/GOD_MODE when local copies are absent."""
    for candidate in candidate_model_asset_paths(
        filename,
        package_dir=package_dir,
        legacy_dir=legacy_dir,
    ):
        if candidate.exists():
            return candidate
    return package_dir / filename

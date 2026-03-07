"""Resolve model and embedding assets from the local asee cache."""

from __future__ import annotations

from pathlib import Path

PACKAGE_MODELS_DIR = Path(__file__).resolve().parent / "models"


def candidate_model_asset_paths(
    filename: str,
    *,
    package_dir: Path = PACKAGE_MODELS_DIR,
) -> tuple[Path, ...]:
    """Return candidate paths for the requested asset inside asee."""
    return (package_dir / filename,)


def resolve_model_asset_path(
    filename: str,
    *,
    package_dir: Path = PACKAGE_MODELS_DIR,
) -> Path:
    """Resolve an asset path inside asee, returning the local cache destination when absent."""
    for candidate in candidate_model_asset_paths(filename, package_dir=package_dir):
        if candidate.exists():
            return candidate
    return package_dir / filename

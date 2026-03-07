"""Persistence helpers for captured face crops."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from time import time
from typing import Any

logger = logging.getLogger(__name__)

type ImageWriter = Callable[[Path, Any], bool]


def _default_write_image(path: Path, image: Any) -> bool:
    import cv2  # type: ignore[import-not-found]

    return bool(cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 90]))


class FaceCaptureWriter:
    """Persist face crops into date-based directories with guard rails."""

    MAX_FILES_PER_DAY = 5_000
    MAX_TOTAL_FILES = 50_000
    MAX_DISK_MB = 2_048
    _CACHE_TTL = 60.0

    def __init__(
        self,
        capture_dir: str | Path,
        min_interval_sec: float = 1.0,
        max_files_per_day: int = MAX_FILES_PER_DAY,
        max_total_files: int = MAX_TOTAL_FILES,
        max_disk_mb: int = MAX_DISK_MB,
        write_image: ImageWriter | None = None,
    ) -> None:
        self._root = Path(capture_dir)
        self._min_interval = min_interval_sec
        self._max_per_day = max_files_per_day
        self._max_total = max_total_files
        self._max_bytes = max_disk_mb * 1024 * 1024
        self._write_image = write_image or _default_write_image

        self._last_saved_at = 0.0
        self._limit_hit = False
        self._cache_time = 0.0
        self._cached_total_files = 0
        self._cached_total_bytes = 0

    def _today_dir(self) -> Path:
        now = datetime.now()
        today_dir = (
            self._root
            / now.strftime("%Y")
            / now.strftime("%m")
            / now.strftime("%d")
            / now.strftime("%H")
            / now.strftime("%M")
        )
        today_dir.mkdir(parents=True, exist_ok=True)
        return today_dir

    def _refresh_cache(self) -> None:
        if time() - self._cache_time < self._CACHE_TTL:
            return
        jpgs = list(self._root.rglob("*.jpg"))
        self._cached_total_files = len(jpgs)
        self._cached_total_bytes = sum(path.stat().st_size for path in jpgs)
        self._cache_time = time()

    def _check_limits(self, today_dir: Path) -> tuple[bool, str]:
        self._refresh_cache()

        today_count = sum(1 for _ in today_dir.glob("*.jpg"))
        if today_count >= self._max_per_day:
            return False, f"本日の保存上限 {today_count}/{self._max_per_day} ファイルに達しました"
        if self._cached_total_files >= self._max_total:
            return (
                False,
                f"総ファイル数上限 {self._cached_total_files}/{self._max_total} に達しました",
            )
        if self._cached_total_bytes >= self._max_bytes:
            current_mb = self._cached_total_bytes // (1024 * 1024)
            max_mb = self._max_bytes // (1024 * 1024)
            return False, f"ディスク使用量上限 {current_mb}/{max_mb} MB に達しました"
        return True, ""

    def save(self, image: Any, score: float) -> bool:
        if image is None:
            return False
        image_size = getattr(image, "size", None)
        if isinstance(image_size, int) and image_size <= 0:
            return False

        now = time()
        if now - self._last_saved_at < self._min_interval:
            return False

        today_dir = self._today_dir()
        ok, reason = self._check_limits(today_dir)
        if not ok:
            if not self._limit_hit:
                logger.warning("[FaceCaptureWriter] 保存停止: %s (%s)", reason, self._root)
                self._limit_hit = True
            return False

        self._limit_hit = False
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        sequence = sum(1 for _ in today_dir.glob("*.jpg")) + 1
        filename = today_dir / f"face-{timestamp}-{sequence:04d}-score{score:.2f}.jpg"

        if not self._write_image(filename, image):
            return False

        self._last_saved_at = now
        self._cached_total_files += 1
        try:
            self._cached_total_bytes += filename.stat().st_size
        except OSError:
            pass
        return True

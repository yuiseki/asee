"""Persistence helpers for periodically sampled full-frame captures."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from time import time
from typing import Any

logger = logging.getLogger(__name__)

type ImageWriter = Callable[[Path, Any], bool]
type DatetimeProvider = Callable[[], datetime]
type TimeProvider = Callable[[], float]


def _default_write_image(path: Path, image: Any) -> bool:
    import cv2

    return bool(cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 90]))


class FullFrameCaptureWriter:
    """Persist sampled full-frame images into camera-specific date partitions."""

    def __init__(
        self,
        capture_dir: str | Path,
        *,
        morning_interval_sec: float = 5 * 60.0,
        day_interval_sec: float = 15 * 60.0,
        evening_interval_sec: float = 10 * 60.0,
        overnight_interval_sec: float = 0.0,
        write_image: ImageWriter | None = None,
        now_provider: DatetimeProvider | None = None,
        time_provider: TimeProvider | None = None,
    ) -> None:
        self._root = Path(capture_dir)
        self._morning_interval_sec = float(morning_interval_sec)
        self._day_interval_sec = float(day_interval_sec)
        self._evening_interval_sec = float(evening_interval_sec)
        self._overnight_interval_sec = float(overnight_interval_sec)
        self._write_image = write_image or _default_write_image
        self._now = now_provider or datetime.now
        self._time = time_provider or time
        self._last_saved_at_by_camera: dict[int, float] = {}

    def interval_for(self, captured_at: datetime) -> tuple[str, float]:
        hour = captured_at.hour
        if 5 <= hour < 10:
            return ("morning", self._morning_interval_sec)
        if 10 <= hour < 17:
            return ("day", self._day_interval_sec)
        if 17 <= hour < 24:
            return ("evening", self._evening_interval_sec)
        return ("overnight", self._overnight_interval_sec)

    def save(
        self,
        image: Any,
        *,
        camera_id: int,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        if image is None:
            return False

        image_size = getattr(image, "size", None)
        if isinstance(image_size, int) and image_size <= 0:
            return False

        captured_at = self._now()
        period, interval_sec = self.interval_for(captured_at)
        if interval_sec <= 0.0:
            return False

        now_ts = self._time()
        last_saved_at = self._last_saved_at_by_camera.get(int(camera_id), 0.0)
        if now_ts - last_saved_at < interval_sec:
            return False

        target_dir = (
            self._root
            / f"video{int(camera_id)}"
            / captured_at.strftime("%Y")
            / captured_at.strftime("%m")
            / captured_at.strftime("%d")
            / captured_at.strftime("%H")
            / captured_at.strftime("%M")
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        sequence = sum(1 for _ in target_dir.glob("*.jpg")) + 1
        timestamp = captured_at.strftime("%Y%m%d-%H%M%S")
        image_path = target_dir / f"frame-{timestamp}-{sequence:04d}.jpg"

        if not self._write_image(image_path, image):
            return False

        sidecar_payload: dict[str, Any] = {
            "capturedAt": captured_at.isoformat(timespec="seconds"),
            "cameraId": int(camera_id),
            "schedulePeriod": period,
        }
        if metadata:
            sidecar_payload.update(metadata)
        image_path.with_suffix(".json").write_text(
            json.dumps(sidecar_payload, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

        self._last_saved_at_by_camera[int(camera_id)] = now_ts
        logger.info(
            "Saved full-frame sample for camera %s (%s) -> %s",
            camera_id,
            period,
            image_path,
        )
        return True

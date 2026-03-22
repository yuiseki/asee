"""InsightFace detector adapter with a YuNet-compatible output contract."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt

type FrameArray = npt.NDArray[np.uint8]
type DetectionRows = npt.NDArray[np.float32]


def _map_insightface_device(device: str) -> tuple[list[str], int]:
    lowered = device.lower()
    if lowered == "cpu":
        return (["CPUExecutionProvider"], -1)
    if lowered.startswith("cuda"):
        if ":" in lowered:
            return (
                ["CUDAExecutionProvider", "CPUExecutionProvider"],
                int(lowered.split(":", 1)[1]),
            )
        return (["CUDAExecutionProvider", "CPUExecutionProvider"], 0)
    raise ValueError(f"unsupported insightface device: {device}")


class InsightFaceDetector:
    """Face detector adapter that mimics cv2.FaceDetectorYN enough for ASEE."""

    def __init__(
        self,
        *,
        device: str = "cuda",
        det_size: int = 320,
        input_size: tuple[int, int] = (1280, 720),
    ) -> None:
        try:
            from insightface.app import FaceAnalysis  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - exercised in runtime only
            raise RuntimeError(
                "insightface is required for InsightFaceDetector. "
                "Install via: pip install insightface onnxruntime-gpu"
            ) from exc

        providers, ctx_id = _map_insightface_device(device)
        self._app = FaceAnalysis(name="buffalo_l", providers=providers)
        self._app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
        self._input_size = input_size
        self._det_size = det_size
        self._providers = tuple(providers)
        self._ctx_id = ctx_id

    @property
    def active_provider(self) -> str:
        if not self._providers:
            return "unknown"
        return self._providers[0]

    @property
    def det_size(self) -> int:
        return self._det_size

    def setInputSize(self, size: tuple[int, int]) -> None:  # noqa: N802
        self._input_size = size

    def set_input_size(self, size: tuple[int, int]) -> None:
        self.setInputSize(size)

    def detect(self, frame: FrameArray) -> tuple[None, DetectionRows | None]:
        _, results = self.detect_batch([frame])
        if not results:
            return (None, None)
        return (None, results[0])

    def detect_batch(
        self,
        frames: Sequence[FrameArray],
    ) -> tuple[None, list[DetectionRows | None]]:
        batch_results: list[DetectionRows | None] = []
        for frame in frames:
            faces = cast(list[Any], self._app.get(frame))
            batch_results.append(self._convert_faces_to_rows(faces))
        return (None, batch_results)

    @staticmethod
    def _convert_faces_to_rows(faces: Sequence[Any]) -> DetectionRows | None:
        if not faces:
            return None
        rows: list[npt.NDArray[np.float32]] = []
        for face in faces:
            bbox = np.asarray(
                face["bbox"] if isinstance(face, dict) else face.bbox,
                dtype=np.float32,
            ).reshape(-1)
            x1 = float(bbox[0])
            y1 = float(bbox[1])
            x2 = float(bbox[2])
            y2 = float(bbox[3])
            row = np.zeros(15, dtype=np.float32)
            row[0] = x1
            row[1] = y1
            row[2] = max(0.0, x2 - x1)
            row[3] = max(0.0, y2 - y1)

            raw_kps = face.get("kps") if isinstance(face, dict) else getattr(face, "kps", None)
            if raw_kps is not None:
                kps = np.asarray(raw_kps, dtype=np.float32).reshape(-1)
                row[4 : 4 + min(10, kps.size)] = kps[:10]

            row[14] = float(
                face.get("det_score", 0.0)
                if isinstance(face, dict)
                else getattr(face, "det_score", 0.0)
            )
            rows.append(row)
        return np.stack(rows, axis=0).astype(np.float32, copy=False)

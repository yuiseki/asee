"""Runtime state that bridges GOD MODE overlay logic to the extracted HTTP shell."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt

from .biometric_status import BiometricStatusTracker, BiometricStatusValue, FaceLike
from .http_app import OverlayTextState, StreamFactory
from .overlay import GodModeOverlay

type FrameArray = npt.NDArray[np.uint8]
type EmbeddingArray = npt.NDArray[np.float32]
type JpegEncoder = Callable[[FrameArray, int], bytes]


class OverlayLike(Protocol):
    """Minimal overlay surface needed by the extracted server runtime."""

    caption: str
    prediction: str
    _owner_embeddings: EmbeddingArray | None

    def set_caption(self, text: str) -> None: ...
    def set_prediction(self, text: str) -> None: ...
    def set_owner_embedding(self, embedding: EmbeddingArray) -> None: ...


class SeeingServerRuntime:
    """State holder for the extracted viewer/backend contract."""

    def __init__(
        self,
        *,
        title: str = "GOD MODE",
        overlay: OverlayLike | None = None,
        camera_ids: Sequence[int] = (),
        jpeg_encoder: JpegEncoder | None = None,
        jpeg_quality: int = 80,
        stream_factory: StreamFactory | None = None,
        owner_embedding_path: str | Path | None = None,
    ) -> None:
        self.title = title
        self.overlay = overlay if overlay is not None else GodModeOverlay()
        self.overlay_state = OverlayTextState(
            caption=self.overlay.caption,
            prediction=self.overlay.prediction,
        )
        self.is_running = False
        self.transport = "webrtc"
        self.camera_ids: Sequence[int] = tuple(int(camera_id) for camera_id in camera_ids)
        self._primary_camera_id = self.camera_ids[0] if self.camera_ids else None
        self._jpeg_encoder = jpeg_encoder
        self._jpeg_quality = jpeg_quality
        self._stream_factory = stream_factory
        self.current_frame: FrameArray | None = None
        self._frames_by_camera: dict[int, FrameArray | None] = {
            camera_id: None for camera_id in self.camera_ids
        }
        self._frame_revision = 0
        self._frame_revisions_by_camera: dict[int, int] = {
            camera_id: 0 for camera_id in self.camera_ids
        }
        self._frame_condition = threading.Condition()
        self._faces: list[FaceLike] = []
        self._faces_by_camera: dict[int, list[FaceLike]] = {
            camera_id: [] for camera_id in self.camera_ids
        }
        tracker_camera_ids = self.camera_ids if self.camera_ids else None
        self._biometric_tracker = BiometricStatusTracker(camera_ids=tracker_camera_ids)

        if owner_embedding_path is not None:
            self.load_owner_embedding(owner_embedding_path)

    @property
    def owner_embedding_loaded(self) -> bool:
        return self.overlay._owner_embeddings is not None

    def set_running(self, running: bool) -> None:
        self.is_running = running

    def update_overlay_text(self, *, caption: str = "", prediction: str = "") -> None:
        self.overlay.set_caption(caption)
        self.overlay.set_prediction(prediction)
        self.overlay_state = OverlayTextState(caption=caption, prediction=prediction)

    def update_frame(self, frame: FrameArray, *, camera_id: int | None = None) -> None:
        with self._frame_condition:
            self._frame_revision += 1
            if self.camera_ids:
                if camera_id is None:
                    raise ValueError("camera_id is required when tracking multiple cameras")
                current_camera_id = int(camera_id)
                self._frames_by_camera[current_camera_id] = frame
                self._frame_revisions_by_camera[current_camera_id] += 1
                if current_camera_id == self._primary_camera_id:
                    self.current_frame = frame
                self._frame_condition.notify_all()
                return

            self.current_frame = frame
            self._frame_condition.notify_all()

    def get_frame(self, camera_id: int | None = None) -> FrameArray | None:
        if camera_id is None or not self.camera_ids:
            return self.current_frame
        return self._frames_by_camera.get(int(camera_id))

    def get_frame_revision(self, camera_id: int | None = None) -> int:
        if camera_id is None or not self.camera_ids:
            return self._frame_revision
        return self._frame_revisions_by_camera.get(int(camera_id), 0)

    def wait_for_frame_update(
        self,
        *,
        camera_id: int | None = None,
        after_revision: int,
        timeout_sec: float,
    ) -> int:
        def current_revision() -> int:
            return self.get_frame_revision(camera_id)

        with self._frame_condition:
            if current_revision() > after_revision:
                return current_revision()
            self._frame_condition.wait_for(
                lambda: current_revision() > after_revision,
                timeout=max(0.0, timeout_sec),
            )
            return current_revision()

    def record_faces(
        self,
        faces: Iterable[FaceLike],
        *,
        camera_id: int | None = None,
        seen_at: float | None = None,
    ) -> None:
        face_list = list(faces)
        if self.camera_ids:
            if camera_id is None:
                raise ValueError("camera_id is required when tracking multiple cameras")
            self._faces_by_camera[int(camera_id)] = face_list
        else:
            self._faces = face_list
        self._biometric_tracker.record_faces(face_list, camera_id=camera_id, seen_at=seen_at)

    def get_faces(self, camera_id: int | None = None) -> list[FaceLike]:
        if camera_id is None or not self.camera_ids:
            return list(self._faces)
        return list(self._faces_by_camera.get(int(camera_id), []))

    def load_owner_embedding(self, path: str | Path) -> None:
        embedding = cast(EmbeddingArray, np.load(Path(path)))
        self.overlay.set_owner_embedding(embedding)

    def get_biometric_status(
        self,
        *,
        now: float | None = None,
    ) -> dict[str, BiometricStatusValue]:
        return self._biometric_tracker.snapshot(
            running=self.is_running,
            owner_embedding_loaded=self.owner_embedding_loaded,
            now=now,
        )

    def get_snapshot_jpeg(self) -> bytes | None:
        if self.current_frame is None or self._jpeg_encoder is None:
            return None
        return self._jpeg_encoder(self.current_frame, self._jpeg_quality)

    def iter_mjpeg(self, device: int | None = None) -> Iterable[bytes] | None:
        if self._stream_factory is None:
            return None
        return self._stream_factory(device)

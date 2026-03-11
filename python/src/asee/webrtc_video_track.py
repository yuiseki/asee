"""Runtime-backed WebRTC video tracks for staged asee transport migration."""

from __future__ import annotations

import asyncio
import fractions
import time

import av
import numpy as np
import numpy.typing as npt
from aiortc import VideoStreamTrack

from .overlay_broadcaster import OverlayBroadcaster
from .overlay_data import FaceDetection, OverlayFrame
from .server_runtime import SeeingServerRuntime

type FrameArray = npt.NDArray[np.uint8]


def _black_frame(width: int = 320, height: int = 240) -> FrameArray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _to_yuv420_frame(frame: FrameArray) -> av.VideoFrame:
    return av.VideoFrame.from_ndarray(frame, format="bgr24").reformat(format="yuv420p")


class RuntimeVideoTrack(VideoStreamTrack):
    """aiortc track that reads frames and face metadata from SeeingServerRuntime."""

    kind = "video"
    _TIME_BASE = fractions.Fraction(1, 90000)

    def __init__(
        self,
        runtime: SeeingServerRuntime,
        *,
        camera_id: int | None,
        fps: int = 10,
        broadcaster: OverlayBroadcaster | None = None,
    ) -> None:
        super().__init__()
        self._runtime = runtime
        self._camera_id = camera_id
        self._fps = max(1, fps)
        self._broadcaster = broadcaster
        self._seq = 0
        self._pts = 0
        self._pts_step = int(90000 / self._fps)
        self._last_frame = _black_frame()
        self._last_revision = self._runtime.get_frame_revision(self._camera_id)

    async def recv(self) -> av.VideoFrame:
        self._last_revision = await asyncio.to_thread(
            self._runtime.wait_for_frame_update,
            camera_id=self._camera_id,
            after_revision=self._last_revision,
            timeout_sec=1.0 / self._fps,
        )

        frame = self._runtime.get_frame(self._camera_id)
        if frame is None:
            frame = self._last_frame
        else:
            self._last_frame = frame

        if self._broadcaster is not None:
            self._broadcaster.broadcast(
                OverlayFrame(
                    seq=self._seq,
                    ts_ms=int(time.time() * 1000),
                    camera_id=0 if self._camera_id is None else self._camera_id,
                    frame_width=int(frame.shape[1]),
                    frame_height=int(frame.shape[0]),
                    caption=self._runtime.overlay_state.caption,
                    prediction=self._runtime.overlay_state.prediction,
                    faces=[
                        FaceDetection(
                            x=int(getattr(face, "x", 0)),
                            y=int(getattr(face, "y", 0)),
                            w=int(getattr(face, "w", 0)),
                            h=int(getattr(face, "h", 0)),
                            label=str(face.label),
                            confidence=float(getattr(face, "confidence", 0.0)),
                        )
                        for face in self._runtime.get_faces(self._camera_id)
                    ],
                )
            )

        self._seq += 1
        yuv_frame = _to_yuv420_frame(frame)
        yuv_frame.pts = self._pts
        yuv_frame.time_base = self._TIME_BASE
        self._pts += self._pts_step
        return yuv_frame

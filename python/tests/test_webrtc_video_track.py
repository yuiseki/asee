"""Tests for runtime-backed WebRTC video tracks."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import numpy as np

from asee.overlay_broadcaster import OverlayBroadcaster
from asee.server_runtime import SeeingServerRuntime
from asee.webrtc_video_track import RuntimeVideoTrack


class FakeOverlay:
    def __init__(self) -> None:
        self.caption = ""
        self.prediction = ""
        self._owner_embeddings: np.ndarray | None = None

    def set_caption(self, text: str) -> None:
        self.caption = text

    def set_prediction(self, text: str) -> None:
        self.prediction = text

    def set_owner_embedding(self, embedding: np.ndarray) -> None:
        self._owner_embeddings = embedding


def test_runtime_video_track_uses_runtime_frame_and_broadcasts_overlay() -> None:
    async def scenario() -> None:
        runtime = SeeingServerRuntime(overlay=FakeOverlay(), camera_ids=(2,))
        frame = np.full((180, 320, 3), 42, dtype=np.uint8)
        runtime.update_frame(frame, camera_id=2)
        runtime.update_overlay_text(caption="OBSERVING", prediction="OWNER PRESENT")
        runtime.record_faces(
            [SimpleNamespace(x=10, y=20, w=30, h=40, label="OWNER", confidence=0.98)],
            camera_id=2,
        )

        payloads: list[str] = []
        channel = SimpleNamespace(
            readyState="open",
            send=lambda payload: payloads.append(payload),
        )
        broadcaster = OverlayBroadcaster()
        broadcaster.add_channel(channel)

        track = RuntimeVideoTrack(runtime, camera_id=2, fps=1000, broadcaster=broadcaster)
        video_frame = await track.recv()

        assert video_frame.width == 320
        assert video_frame.height == 180
        assert len(payloads) == 1
        assert '"camera_id": 2' in payloads[0]
        assert '"frame_width": 320' in payloads[0]
        assert '"frame_height": 180' in payloads[0]
        assert '"label": "OWNER"' in payloads[0]
        assert '"caption": "OBSERVING"' in payloads[0]

    asyncio.run(scenario())


def test_runtime_video_track_returns_soon_after_new_frame_arrives() -> None:
    async def scenario() -> None:
        runtime = SeeingServerRuntime(overlay=FakeOverlay(), camera_ids=(2,))
        track = RuntimeVideoTrack(runtime, camera_id=2, fps=10, broadcaster=None)

        async def push_frame() -> None:
            await asyncio.sleep(0.01)
            runtime.update_frame(np.full((120, 160, 3), 7, dtype=np.uint8), camera_id=2)

        start = time.perf_counter()
        producer = asyncio.create_task(push_frame())
        video_frame = await track.recv()
        elapsed = time.perf_counter() - start
        await producer

        assert video_frame.width == 160
        assert video_frame.height == 120
        assert elapsed < 0.08

    asyncio.run(scenario())

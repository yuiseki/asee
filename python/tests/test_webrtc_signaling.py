"""Tests for the extracted WebRTC signaling app."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
from aiohttp.test_utils import TestClient, TestServer

from asee.overlay_broadcaster import OverlayBroadcaster
from asee.server_runtime import SeeingServerRuntime
from asee.webrtc_signaling import create_webrtc_app


async def _make_client(app: Any) -> Any:
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    return client


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


def test_status_and_compat_endpoints() -> None:
    async def scenario() -> None:
        app = create_webrtc_app(
            camera_ids=[0, 2, 4, 6],
            overlay_text={"caption": "OBSERVING", "prediction": "ROOM CALM"},
            biometric_status={
                "running": True,
                "ownerEmbeddingLoaded": True,
                "ownerPresent": True,
                "ownerCount": 1,
                "subjectCount": 0,
                "peopleCount": 1,
                "ownerSeenAgoMs": 12,
                "updatedAt": 123.4,
            },
        )
        client = await _make_client(app)
        try:
            cameras = await (await client.get("/cameras")).json()
            status = await (await client.get("/status")).json()
            overlay = await (await client.get("/overlay_text")).json()
            biometric = await (await client.get("/biometric_status")).json()

            assert cameras == {"cameras": [0, 2, 4, 6]}
            assert status["running"] is True
            assert overlay == {"caption": "OBSERVING", "prediction": "ROOM CALM"}
            assert biometric["ownerPresent"] is True
        finally:
            await client.close()

    asyncio.run(scenario())


def test_runtime_backed_endpoints_reflect_live_runtime_state() -> None:
    async def scenario() -> None:
        runtime = SeeingServerRuntime(overlay=FakeOverlay(), camera_ids=(0, 2))
        runtime.set_running(True)
        runtime.update_overlay_text(caption="OBSERVING", prediction="ROOM CALM")
        runtime.record_faces(
            [SimpleNamespace(label="OWNER")],
            camera_id=0,
            seen_at=100.0,
        )
        app = create_webrtc_app(runtime=runtime)
        client = await _make_client(app)
        try:
            cameras = await (await client.get("/cameras")).json()
            status = await (await client.get("/status")).json()
            overlay = await (await client.get("/overlay_text")).json()
            biometric = await (await client.get("/biometric_status")).json()

            assert cameras == {"cameras": [0, 2]}
            assert status["running"] is True
            assert overlay == {"caption": "OBSERVING", "prediction": "ROOM CALM"}
            assert biometric["ownerPresent"] is True
        finally:
            await client.close()

    asyncio.run(scenario())


@patch("asee.webrtc_signaling.RTCPeerConnection")
def test_offer_registers_overlay_datachannel(
    mock_pc_cls: Any,
) -> None:
    async def scenario() -> None:
        handlers: dict[str, Any] = {}
        channel = MagicMock()
        channel.label = "overlay"
        channel.readyState = "open"

        mock_pc = AsyncMock()
        mock_pc.localDescription = MagicMock(type="answer", sdp="v=0\r\n")
        mock_pc.createAnswer = AsyncMock(return_value=MagicMock(type="answer", sdp="v=0\r\n"))
        mock_pc.setLocalDescription = AsyncMock()
        mock_pc.setRemoteDescription = AsyncMock()
        mock_pc.addTrack = MagicMock()
        mock_pc.on = MagicMock(side_effect=lambda event, cb: handlers.setdefault(event, cb))
        mock_pc_cls.return_value = mock_pc

        broadcaster = OverlayBroadcaster()
        app = create_webrtc_app(camera_ids=[0], broadcaster=broadcaster)
        client = await _make_client(app)
        try:
            response = await client.post("/offer", json={"type": "offer", "sdp": "v=0\r\n"})

            assert response.status == 200
            handlers["datachannel"](channel)
            assert broadcaster.channel_count == 1
        finally:
            await client.close()

    asyncio.run(scenario())


@patch("asee.webrtc_signaling.RTCPeerConnection")
def test_runtime_offer_adds_one_track_per_camera(
    mock_pc_cls: Any,
) -> None:
    async def scenario() -> None:
        handlers: dict[str, Any] = {}
        mock_pc = AsyncMock()
        mock_pc.localDescription = MagicMock(type="answer", sdp="v=0\r\n")
        mock_pc.createAnswer = AsyncMock(return_value=MagicMock(type="answer", sdp="v=0\r\n"))
        mock_pc.setLocalDescription = AsyncMock()
        mock_pc.setRemoteDescription = AsyncMock()
        mock_pc.addTrack = MagicMock()
        mock_pc.on = MagicMock(side_effect=lambda event, cb: handlers.setdefault(event, cb))
        mock_pc_cls.return_value = mock_pc

        runtime = SeeingServerRuntime(overlay=FakeOverlay(), camera_ids=(0, 2, 4, 6))
        app = create_webrtc_app(runtime=runtime, broadcaster=OverlayBroadcaster())
        client = await _make_client(app)
        try:
            response = await client.post("/offer", json={"type": "offer", "sdp": "v=0\r\n"})

            assert response.status == 200
            assert mock_pc.addTrack.call_count == 4
        finally:
            await client.close()

    asyncio.run(scenario())

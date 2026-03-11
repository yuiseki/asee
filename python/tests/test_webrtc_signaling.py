"""Tests for the extracted WebRTC signaling app."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from aiohttp.test_utils import TestClient, TestServer

from asee.overlay_broadcaster import OverlayBroadcaster
from asee.webrtc_signaling import create_webrtc_app


async def _make_client(app: Any) -> TestClient:
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    return client


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

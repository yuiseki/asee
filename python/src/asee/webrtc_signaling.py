"""WebRTC signaling app for the asee migration path."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription

from .overlay_broadcaster import OverlayBroadcaster

type VideoTrackFactory = Callable[[], Any]
type VideoTracksFactory = Callable[[], list[Any]]

_peer_connections: set[RTCPeerConnection] = set()


def _default_biometric_status() -> dict[str, bool | int | float | None]:
    return {
        "running": True,
        "ownerEmbeddingLoaded": False,
        "ownerPresent": False,
        "ownerCount": 0,
        "subjectCount": 0,
        "peopleCount": 0,
        "ownerSeenAgoMs": None,
        "updatedAt": 0.0,
    }


async def _handle_offer(
    request: web.Request,
    *,
    video_track_factory: VideoTrackFactory | None,
    video_tracks_factory: VideoTracksFactory | None,
    broadcaster: OverlayBroadcaster | None,
) -> web.Response:
    content_type = request.headers.get("Content-Type", "")
    if "application/json" not in content_type:
        raise web.HTTPUnsupportedMediaType()

    try:
        payload = await request.json()
    except Exception as exc:
        raise web.HTTPBadRequest(reason="Invalid JSON") from exc

    if "sdp" not in payload:
        raise web.HTTPBadRequest(reason="Missing 'sdp' field")
    if "type" not in payload:
        raise web.HTTPBadRequest(reason="Missing 'type' field")

    offer = RTCSessionDescription(sdp=payload["sdp"], type=payload["type"])
    peer = RTCPeerConnection()
    _peer_connections.add(peer)

    def on_connection_state_change() -> None:
        if peer.connectionState not in ("failed", "closed"):
            return

        async def close_peer() -> None:
            await peer.close()
            _peer_connections.discard(peer)

        asyncio.get_running_loop().create_task(close_peer())

    peer.on("connectionstatechange", on_connection_state_change)

    if video_tracks_factory is not None:
        for track in video_tracks_factory():
            peer.addTrack(track)
    elif video_track_factory is not None:
        peer.addTrack(video_track_factory())

    if broadcaster is not None:

        def on_datachannel(channel: Any) -> None:
            if getattr(channel, "label", "") != "overlay":
                return
            broadcaster.add_channel(channel)

            def on_close() -> None:
                broadcaster.remove_channel(channel)

            channel.on("close", on_close)

        peer.on("datachannel", on_datachannel)

    await peer.setRemoteDescription(offer)
    answer = await peer.createAnswer()
    await peer.setLocalDescription(answer)

    return web.json_response(
        {
            "type": peer.localDescription.type,
            "sdp": peer.localDescription.sdp,
        }
    )


def create_webrtc_app(
    *,
    camera_ids: Sequence[int],
    overlay_text: dict[str, str] | None = None,
    biometric_status: dict[str, bool | int | float | None] | None = None,
    broadcaster: OverlayBroadcaster | None = None,
    video_track_factory: VideoTrackFactory | None = None,
    video_tracks_factory: VideoTracksFactory | None = None,
    static_dir: str | Path | None = None,
) -> web.Application:
    app = web.Application()
    camera_ids_payload = list(camera_ids)
    overlay_text_payload = overlay_text or {"caption": "", "prediction": ""}
    biometric_status_payload = biometric_status or _default_biometric_status()

    async def handle_index(_request: web.Request) -> web.Response:
        return web.Response(content_type="text/html", text="<html><body>ASEE WebRTC</body></html>")

    async def handle_status(_request: web.Request) -> web.Response:
        return web.json_response({"running": True, "connections": len(_peer_connections)})

    async def handle_cameras(_request: web.Request) -> web.Response:
        return web.json_response({"cameras": list(camera_ids_payload)})

    async def handle_overlay_text(_request: web.Request) -> web.Response:
        return web.json_response(dict(overlay_text_payload))

    async def handle_biometric_status(_request: web.Request) -> web.Response:
        return web.json_response(dict(biometric_status_payload))

    async def offer_handler(request: web.Request) -> web.Response:
        return await _handle_offer(
            request,
            video_track_factory=video_track_factory,
            video_tracks_factory=video_tracks_factory,
            broadcaster=broadcaster,
        )

    app.router.add_get("/", handle_index)
    app.router.add_post("/offer", offer_handler)
    app.router.add_get("/status", handle_status)
    app.router.add_get("/cameras", handle_cameras)
    app.router.add_get("/overlay_text", handle_overlay_text)
    app.router.add_get("/biometric_status", handle_biometric_status)

    if static_dir is not None:
        static_path = Path(static_dir)
        if static_path.exists():
            app.router.add_static("/static", static_path)

    return app

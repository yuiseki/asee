"""Tests for WebRTC overlay broadcasting."""

from __future__ import annotations

from types import SimpleNamespace

from asee.overlay_broadcaster import OverlayBroadcaster
from asee.overlay_data import OverlayFrame


def test_broadcast_sends_json_to_all_open_channels() -> None:
    sent: list[str] = []

    class Channel:
        def __init__(self) -> None:
            self.readyState = "open"

        def send(self, payload: str) -> None:
            sent.append(payload)

    broadcaster = OverlayBroadcaster()
    broadcaster.add_channel(Channel())
    broadcaster.add_channel(Channel())

    broadcaster.broadcast(OverlayFrame(seq=1, ts_ms=2, faces=[]))

    assert len(sent) == 2
    assert all('"seq": 1' in payload for payload in sent)


def test_broadcast_drops_closed_channels() -> None:
    alive = SimpleNamespace(readyState="open", send=lambda _payload: None)
    dead = SimpleNamespace(readyState="closed", send=lambda _payload: None)

    broadcaster = OverlayBroadcaster()
    broadcaster.add_channel(alive)
    broadcaster.add_channel(dead)

    broadcaster.broadcast(OverlayFrame(seq=1, ts_ms=2, faces=[]))

    assert broadcaster.channel_count == 1

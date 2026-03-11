"""Broadcast overlay JSON payloads to active WebRTC DataChannels."""

from __future__ import annotations

import logging
from typing import Any

from .overlay_data import OverlayFrame

logger = logging.getLogger(__name__)


class OverlayBroadcaster:
    """Small channel registry used by WebRTC signaling handlers."""

    def __init__(self) -> None:
        self._channels: list[Any] = []

    @property
    def channel_count(self) -> int:
        return len(self._channels)

    def add_channel(self, channel: Any) -> None:
        self._channels.append(channel)

    def remove_channel(self, channel: Any) -> None:
        try:
            self._channels.remove(channel)
        except ValueError:
            pass

    def broadcast(self, frame: OverlayFrame) -> None:
        payload = frame.to_json()
        dead_channels: list[Any] = []
        for channel in self._channels:
            if getattr(channel, "readyState", "open") != "open":
                dead_channels.append(channel)
                continue
            try:
                channel.send(payload)
            except Exception as exc:
                logger.warning("DataChannel send failed: %s", exc)
                dead_channels.append(channel)
        for channel in dead_channels:
            self.remove_channel(channel)

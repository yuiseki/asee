"""Application-level WebRTC codec tuning for asee."""

from __future__ import annotations

from typing import Protocol

from aiortc.codecs import h264, vpx

DEFAULT_WEBRTC_VIDEO_BITRATE_BPS = 2_500_000


class _CodecModule(Protocol):
    DEFAULT_BITRATE: int
    MIN_BITRATE: int
    MAX_BITRATE: int


def _apply_codec_default_bitrate(module: _CodecModule, bitrate_bps: int) -> None:
    normalized_bitrate = int(bitrate_bps)
    if normalized_bitrate <= 0:
        raise ValueError("WebRTC video bitrate must be positive")

    current_max = int(module.MAX_BITRATE)
    current_min = int(module.MIN_BITRATE)
    module.MAX_BITRATE = max(current_max, normalized_bitrate)
    module.DEFAULT_BITRATE = max(current_min, normalized_bitrate)


def apply_webrtc_video_tuning(*, video_bitrate_bps: int) -> None:
    """Raise aiortc encoder defaults for the local WebRTC transport."""
    _apply_codec_default_bitrate(vpx, video_bitrate_bps)
    _apply_codec_default_bitrate(h264, video_bitrate_bps)

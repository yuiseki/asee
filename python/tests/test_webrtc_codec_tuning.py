from __future__ import annotations

from aiortc.codecs import h264, vpx

from asee.webrtc_codec_tuning import (
    DEFAULT_WEBRTC_VIDEO_BITRATE_BPS,
    apply_webrtc_video_tuning,
)


def test_apply_webrtc_video_tuning_raises_encoder_defaults(monkeypatch) -> None:
    monkeypatch.setattr(vpx, "DEFAULT_BITRATE", 500_000)
    monkeypatch.setattr(vpx, "MAX_BITRATE", 1_500_000)
    monkeypatch.setattr(vpx, "MIN_BITRATE", 250_000)
    monkeypatch.setattr(h264, "DEFAULT_BITRATE", 1_000_000)
    monkeypatch.setattr(h264, "MAX_BITRATE", 3_000_000)
    monkeypatch.setattr(h264, "MIN_BITRATE", 500_000)

    apply_webrtc_video_tuning(video_bitrate_bps=DEFAULT_WEBRTC_VIDEO_BITRATE_BPS)

    assert vpx.DEFAULT_BITRATE == DEFAULT_WEBRTC_VIDEO_BITRATE_BPS
    assert vpx.MAX_BITRATE == DEFAULT_WEBRTC_VIDEO_BITRATE_BPS
    assert h264.DEFAULT_BITRATE == DEFAULT_WEBRTC_VIDEO_BITRATE_BPS
    assert h264.MAX_BITRATE == 3_000_000

"""Overlay payload types for WebRTC DataChannel delivery."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class FaceDetection:
    """A single detected face sent to the WebRTC viewer overlay."""

    x: int
    y: int
    w: int
    h: int
    label: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FaceDetection:
        return cls(
            x=int(payload["x"]),
            y=int(payload["y"]),
            w=int(payload["w"]),
            h=int(payload["h"]),
            label=str(payload["label"]),
            confidence=float(payload["confidence"]),
        )


@dataclass(slots=True)
class OverlayFrame:
    """Overlay data for one rendered frame."""

    seq: int
    ts_ms: int
    faces: list[FaceDetection]
    camera_id: int = 0
    caption: str | None = None
    prediction: str | None = None

    def to_json(self) -> str:
        payload: dict[str, Any] = {
            "seq": self.seq,
            "ts_ms": self.ts_ms,
            "camera_id": self.camera_id,
            "faces": [face.to_dict() for face in self.faces],
        }
        if self.caption is not None:
            payload["caption"] = self.caption
        if self.prediction is not None:
            payload["prediction"] = self.prediction
        return json.dumps(payload)

    @classmethod
    def from_json(cls, payload: str) -> OverlayFrame:
        data = json.loads(payload)
        return cls(
            seq=int(data["seq"]),
            ts_ms=int(data["ts_ms"]),
            faces=[FaceDetection.from_dict(face) for face in data.get("faces", [])],
            camera_id=int(data.get("camera_id", 0)),
            caption=data.get("caption"),
            prediction=data.get("prediction"),
        )

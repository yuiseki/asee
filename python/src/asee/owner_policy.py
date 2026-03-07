"""OWNER selection and similarity policy extracted from GOD MODE overlay code."""

from __future__ import annotations

from .tracking import FaceBox

OWNER_TOPK = 20
OWNER_COSINE_THRESHOLD = 0.50


def keep_largest_owner(faces: list[FaceBox]) -> list[FaceBox]:
    """Keep at most one OWNER per camera while preserving SUBJECT faces."""
    owners = [face for face in faces if face.label == "OWNER"]
    if len(owners) <= 1:
        return faces

    largest_owner = max(owners, key=lambda face: face.w * face.h)
    return [face for face in faces if face.label != "OWNER"] + [largest_owner]

"""OWNER selection and similarity policy extracted from GOD MODE overlay code."""

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np
import numpy.typing as npt

from .tracking import FaceBox

OWNER_TOPK = 20
OWNER_COSINE_THRESHOLD = 0.50

type EmbeddingArray = npt.NDArray[np.float32]


class FaceRecognizerLike(Protocol):
    """Minimal FaceRecognizerSF-compatible matching surface."""

    def match(self, reference: np.ndarray, embedding: np.ndarray, metric: int) -> float: ...


def keep_largest_owner(faces: list[FaceBox]) -> list[FaceBox]:
    """Keep at most one OWNER per camera while preserving SUBJECT faces."""
    owners = [face for face in faces if face.label == "OWNER"]
    if len(owners) <= 1:
        return faces

    largest_owner = max(owners, key=lambda face: face.w * face.h)
    return [face for face in faces if face.label != "OWNER"] + [largest_owner]


def classify_owner_embedding(
    *,
    recognizer: FaceRecognizerLike | None,
    owner_embeddings: EmbeddingArray | None,
    embedding: EmbeddingArray,
    face_confidence: float,
) -> tuple[str, float]:
    """Classify one embedding as OWNER or SUBJECT using current owner refs."""
    if recognizer is None or owner_embeddings is None:
        return "SUBJECT", face_confidence

    try:
        scores = sorted(
            [
                float(
                    recognizer.match(
                        reference.reshape(1, -1),
                        embedding,
                        cv2.FaceRecognizerSF_FR_COSINE,
                    )
                )
                for reference in owner_embeddings
            ],
            reverse=True,
        )
        score = float(np.mean(scores[:OWNER_TOPK]))
        if score >= OWNER_COSINE_THRESHOLD:
            return "OWNER", score
        return "SUBJECT", score
    except Exception:
        return "SUBJECT", face_confidence

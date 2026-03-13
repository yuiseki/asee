"""Unit tests for OWNER selection policy."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from asee.owner_policy import classify_owner_embedding, keep_largest_owner
from asee.tracking import FaceBox


def test_keep_largest_owner_leaves_subjects_untouched():
    faces = [
        FaceBox(x=10, y=10, w=50, h=50, label='OWNER'),
        FaceBox(x=100, y=100, w=110, h=110, label='OWNER'),
        FaceBox(x=300, y=120, w=80, h=80, label='SUBJECT'),
    ]

    result = keep_largest_owner(faces)

    owners = [face for face in result if face.label == 'OWNER']
    subjects = [face for face in result if face.label == 'SUBJECT']
    assert len(owners) == 1
    assert owners[0].w == 110
    assert len(subjects) == 1


def test_keep_largest_owner_returns_original_when_zero_or_one_owner():
    subject_only = [FaceBox(x=10, y=10, w=50, h=50, label='SUBJECT')]
    single_owner = [FaceBox(x=10, y=10, w=50, h=50, label='OWNER')]

    assert keep_largest_owner(subject_only) == subject_only
    assert keep_largest_owner(single_owner) == single_owner


class FakeRecognizer:
    def __init__(self, scores: list[float]) -> None:
        self._scores = list(scores)

    def match(self, reference: np.ndarray, embedding: np.ndarray, metric: int) -> float:
        del reference, embedding
        assert metric == cv2.FaceRecognizerSF_FR_COSINE
        return self._scores.pop(0)


def test_classify_owner_embedding_returns_owner_for_high_mean_score():
    recognizer = FakeRecognizer([0.8, 0.7, 0.6])
    owner_embeddings = np.zeros((3, 1, 128), dtype=np.float32)
    embedding = np.zeros((1, 128), dtype=np.float32)

    label, score = classify_owner_embedding(
        recognizer=recognizer,
        owner_embeddings=owner_embeddings,
        embedding=embedding,
        face_confidence=0.2,
    )

    assert label == "OWNER"
    assert score == pytest.approx(0.7)


def test_classify_owner_embedding_returns_subject_below_threshold():
    recognizer = FakeRecognizer([0.4, 0.45, 0.49])
    owner_embeddings = np.zeros((3, 1, 128), dtype=np.float32)
    embedding = np.zeros((1, 128), dtype=np.float32)

    label, score = classify_owner_embedding(
        recognizer=recognizer,
        owner_embeddings=owner_embeddings,
        embedding=embedding,
        face_confidence=0.91,
    )

    assert label == "SUBJECT"
    assert score == pytest.approx((0.49 + 0.45 + 0.4) / 3)


def test_classify_owner_embedding_falls_back_to_face_confidence_on_error():
    class ErrorRecognizer:
        def match(self, reference: np.ndarray, embedding: np.ndarray, metric: int) -> float:
            del reference, embedding, metric
            raise RuntimeError("boom")

    owner_embeddings = np.zeros((1, 1, 128), dtype=np.float32)
    embedding = np.zeros((1, 128), dtype=np.float32)

    label, score = classify_owner_embedding(
        recognizer=ErrorRecognizer(),
        owner_embeddings=owner_embeddings,
        embedding=embedding,
        face_confidence=0.77,
    )

    assert label == "SUBJECT"
    assert score == 0.77

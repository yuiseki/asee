"""Unit tests for OWNER selection policy."""

from __future__ import annotations

from asee.owner_policy import keep_largest_owner
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

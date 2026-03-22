from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from asee.insightface_detector import InsightFaceDetector


def test_convert_faces_to_rows_preserves_bbox_keypoints_and_score() -> None:
    bbox = np.array([10.0, 20.0, 50.0, 80.0], dtype=np.float32)
    kps = np.array(
        [
            [11.0, 21.0],
            [12.0, 22.0],
            [13.0, 23.0],
            [14.0, 24.0],
            [15.0, 25.0],
        ],
        dtype=np.float32,
    )
    face = SimpleNamespace(bbox=bbox, kps=kps, det_score=0.875)

    rows = InsightFaceDetector._convert_faces_to_rows([face])

    assert rows is not None
    assert rows.shape == (1, 15)
    row = rows[0]
    assert row[0] == 10.0
    assert row[1] == 20.0
    assert row[2] == 40.0
    assert row[3] == 60.0
    assert row[4:14].tolist() == [11.0, 21.0, 12.0, 22.0, 13.0, 23.0, 14.0, 24.0, 15.0, 25.0]
    assert row[14] == 0.875


def test_convert_faces_to_rows_returns_none_for_empty_input() -> None:
    assert InsightFaceDetector._convert_faces_to_rows([]) is None


def test_convert_faces_to_rows_zero_fills_missing_keypoints() -> None:
    bbox = np.array([1.0, 2.0, 5.0, 8.0], dtype=np.float32)
    face = {"bbox": bbox, "det_score": 0.5}

    rows = InsightFaceDetector._convert_faces_to_rows([face])

    assert rows is not None
    assert rows.shape == (1, 15)
    assert rows[0][4:14].tolist() == [0.0] * 10

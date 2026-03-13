from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from asee.retrain_owner_embedding import (
    DatasetEvaluation,
    GreedySelectionResult,
    augment_owner_embeddings,
    evaluate_image_paths,
    greedy_select_false_negative_candidates,
    iter_image_paths,
    normalize_owner_embeddings,
    resolve_latest_guest_session_dir,
    snapshot_owner_embedding,
)


def test_normalize_owner_embeddings_accepts_supported_shapes() -> None:
    one_dim = np.zeros(128, dtype=np.float32)
    two_dim = np.zeros((2, 128), dtype=np.float32)
    three_dim = np.zeros((3, 1, 128), dtype=np.float32)

    assert normalize_owner_embeddings(one_dim).shape == (1, 1, 128)
    assert normalize_owner_embeddings(two_dim).shape == (2, 1, 128)
    assert normalize_owner_embeddings(three_dim).shape == (3, 1, 128)


def test_augment_owner_embeddings_concatenates_normalized_arrays() -> None:
    current = np.zeros((2, 1, 128), dtype=np.float32)
    additions = np.ones((3, 128), dtype=np.float32)

    combined = augment_owner_embeddings(current=current, additions=additions)

    assert combined.shape == (5, 1, 128)
    assert np.all(combined[:2] == 0)
    assert np.all(combined[2:] == 1)


def test_snapshot_owner_embedding_copies_file_into_snapshot_dir(tmp_path: Path) -> None:
    source = tmp_path / "owner_embedding.npy"
    snapshot_dir = tmp_path / "snapshots"
    expected = np.zeros((2, 1, 128), dtype=np.float32)
    np.save(source, expected)

    snapshot = snapshot_owner_embedding(
        owner_embedding_path=source,
        snapshot_dir=snapshot_dir,
        timestamp="2026-03-13_23-00-00",
    )

    assert snapshot == snapshot_dir / "owner_embedding_2026-03-13_23-00-00.npy"
    assert np.array_equal(np.load(snapshot), expected)


def test_resolve_latest_guest_session_dir_returns_latest_directory(tmp_path: Path) -> None:
    first = tmp_path / "2026-03-13_20-00-00"
    second = tmp_path / "2026-03-13_22-41-56"
    first.mkdir(parents=True)
    second.mkdir(parents=True)

    assert resolve_latest_guest_session_dir(tmp_path) == second


def test_iter_image_paths_filters_out_sidecar_json_files(tmp_path: Path) -> None:
    image = tmp_path / "face.jpg"
    sidecar = tmp_path / "face.json"
    image.write_bytes(b"x")
    sidecar.write_text("{}", encoding="utf-8")

    assert iter_image_paths(tmp_path) == [image]


def test_evaluate_image_paths_counts_owner_subject_and_skipped(tmp_path: Path) -> None:
    image_paths = [
        tmp_path / "owner-hit.jpg",
        tmp_path / "subject-hit.jpg",
        tmp_path / "skip.jpg",
    ]
    for path in image_paths:
        path.write_bytes(b"x")

    def read_image(path: Path) -> np.ndarray | None:
        if path.name == "skip.jpg":
            return None
        return np.ones((16, 16, 3), dtype=np.uint8)

    def extract_embedding(frame: np.ndarray) -> np.ndarray | None:
        del frame
        return np.ones((1, 128), dtype=np.float32)

    def classify_embedding(path: Path, embedding: np.ndarray) -> tuple[str, float]:
        del embedding
        if path.name == "owner-hit.jpg":
            return "OWNER", 0.9
        return "SUBJECT", 0.3

    evaluation = evaluate_image_paths(
        image_paths=image_paths,
        read_image=read_image,
        extract_embedding=extract_embedding,
        classify_embedding=classify_embedding,
    )

    assert evaluation == DatasetEvaluation(
        total_files=3,
        usable_embeddings=2,
        skipped_files=1,
        owner_predictions=1,
        subject_predictions=1,
        mean_score=pytest.approx(0.6),
    )


def test_greedy_select_false_negative_candidates_prefers_positive_gain_without_negative_harm(
) -> None:
    positive_candidate_scores = np.array(
        [
            [0.9, 0.9],
            [0.7, 0.4],
        ],
        dtype=np.float32,
    )
    negative_candidate_scores = np.array(
        [
            [0.2, 0.2],
            [0.9, 0.9],
        ],
        dtype=np.float32,
    )
    positive_topk = np.array([[0.4], [0.4]], dtype=np.float32)
    negative_topk = np.array([[0.4], [0.4]], dtype=np.float32)
    candidate_paths = [Path("good.jpg"), Path("bad.jpg")]

    result = greedy_select_false_negative_candidates(
        candidate_paths=candidate_paths,
        positive_candidate_scores=positive_candidate_scores,
        negative_candidate_scores=negative_candidate_scores,
        positive_topk_values=positive_topk,
        negative_topk_values=negative_topk,
        threshold=0.5,
        negative_penalty=3.0,
    )

    assert result == GreedySelectionResult(
        selected_indices=(0,),
        selected_paths=(Path("good.jpg"),),
    )


def test_greedy_select_false_negative_candidates_stops_when_no_candidate_has_positive_utility(
) -> None:
    positive_candidate_scores = np.array([[0.49, 0.49]], dtype=np.float32)
    negative_candidate_scores = np.array([[0.9, 0.9]], dtype=np.float32)
    positive_topk = np.array([[0.48], [0.48]], dtype=np.float32)
    negative_topk = np.array([[0.48], [0.48]], dtype=np.float32)
    candidate_paths = [Path("harmful.jpg")]

    result = greedy_select_false_negative_candidates(
        candidate_paths=candidate_paths,
        positive_candidate_scores=positive_candidate_scores,
        negative_candidate_scores=negative_candidate_scores,
        positive_topk_values=positive_topk,
        negative_topk_values=negative_topk,
        threshold=0.5,
        negative_penalty=3.0,
    )

    assert result == GreedySelectionResult(
        selected_indices=(),
        selected_paths=(),
    )

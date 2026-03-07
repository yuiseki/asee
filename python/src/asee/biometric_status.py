"""Biometric status aggregation independent from camera / HTTP runtime details."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from time import time
from typing import Protocol


class FaceLike(Protocol):
    """Minimal interface needed for biometric aggregation."""

    label: str


type BiometricStatusValue = bool | int | float | None


@dataclass(slots=True)
class BiometricStatusTracker:
    """Track OWNER presence and aggregate counts across one or more cameras."""

    camera_ids: Iterable[int] | None = None
    _faces: list[FaceLike] = field(default_factory=list, init=False, repr=False)
    _faces_by_camera: dict[int, list[FaceLike]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _owner_last_seen_at: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.camera_ids is None:
            return
        self._faces_by_camera = {int(camera_id): [] for camera_id in self.camera_ids}

    def record_faces(
        self,
        faces: Iterable[FaceLike],
        *,
        camera_id: int | None = None,
        seen_at: float | None = None,
    ) -> None:
        """Replace the visible faces for one camera (or the single-camera runtime)."""
        face_list = list(faces)

        if self._faces_by_camera:
            if camera_id is None:
                raise ValueError("camera_id is required when tracking multiple cameras")
            self._faces_by_camera[int(camera_id)] = face_list
        else:
            self._faces = face_list

        if any(getattr(face, "label", "") == "OWNER" for face in face_list):
            self._owner_last_seen_at = seen_at if seen_at is not None else time()

    def snapshot(
        self,
        *,
        running: bool,
        owner_embedding_loaded: bool,
        now: float | None = None,
    ) -> dict[str, BiometricStatusValue]:
        """Return the GOD MODE-compatible biometric status payload."""
        current_time = now if now is not None else time()
        faces = self._aggregate_faces()
        owner_count = sum(1 for face in faces if getattr(face, "label", "") == "OWNER")
        subject_count = sum(1 for face in faces if getattr(face, "label", "") != "OWNER")

        owner_seen_ago_ms: int | None = None
        if self._owner_last_seen_at is not None:
            owner_seen_ago_ms = max(0, int((current_time - self._owner_last_seen_at) * 1000))

        return {
            "running": running,
            "ownerEmbeddingLoaded": owner_embedding_loaded,
            "ownerPresent": owner_count > 0,
            "ownerCount": owner_count,
            "subjectCount": subject_count,
            "peopleCount": owner_count + subject_count,
            "ownerSeenAgoMs": owner_seen_ago_ms,
            "updatedAt": current_time,
        }

    def _aggregate_faces(self) -> list[FaceLike]:
        if self._faces_by_camera:
            aggregated: list[FaceLike] = []
            for faces in self._faces_by_camera.values():
                aggregated.extend(faces)
            return aggregated
        return list(self._faces)

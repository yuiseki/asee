"""Face tracking primitives extracted from GOD MODE overlay code."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import hypot
from typing import Any


@dataclass(slots=True)
class FaceBox:
    """Detected face rectangle with recognition metadata."""

    x: int
    y: int
    w: int
    h: int
    label: str = "SUBJECT"
    confidence: float = 1.0
    raw_detection: Any = field(default=None, repr=False)
    id: int = 0

    @classmethod
    def from_yunet_row(cls, row: Sequence[float]) -> FaceBox:
        """Create a FaceBox from a YuNet detection row.

        YuNet output format: [x, y, w, h, kps_x1, kps_y1, ..., kps_x5, kps_y5, score]
        Row has 15 elements; score is at index 14.
        """
        return cls(
            x=int(row[0]),
            y=int(row[1]),
            w=int(row[2]),
            h=int(row[3]),
            confidence=float(row[14]),
            raw_detection=row,
        )

    def corners(self) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
        return (
            (self.x, self.y),
            (self.x + self.w, self.y),
            (self.x, self.y + self.h),
            (self.x + self.w, self.y + self.h),
        )

    def iou(self, other: FaceBox) -> float:
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.w, other.x + other.w)
        y2 = min(self.y + self.h, other.y + other.h)

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = self.w * self.h
        area2 = other.w * other.h
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0


class FaceTracker:
    """Track and smooth face boxes over time."""

    def __init__(self, alpha: float = 0.4, max_lost_frames: int = 2, min_hits: int = 3):
        self.alpha = alpha
        self.max_lost_frames = max_lost_frames
        self.min_hits = min_hits
        self.tracked_boxes: list[FaceBox] = []
        self._lost_counts: list[int] = []
        self._hit_counts: list[int] = []
        self._next_id = 1

    def update(self, detected_boxes: list[FaceBox]) -> list[FaceBox]:
        if not self.tracked_boxes:
            for face_box in detected_boxes:
                face_box.id = self._next_id
                self._next_id += 1
                self.tracked_boxes.append(face_box)
                self._lost_counts.append(0)
                self._hit_counts.append(1)
            return self._confirmed_tracks()

        matched_indices = [-1] * len(detected_boxes)
        used_tracks = [False] * len(self.tracked_boxes)

        for detected_index, detected_box in enumerate(detected_boxes):
            best_score = -1.0
            best_track_index = -1
            for track_index, tracked_box in enumerate(self.tracked_boxes):
                if used_tracks[track_index]:
                    continue

                iou = detected_box.iou(tracked_box)
                dist = hypot(
                    (detected_box.x + detected_box.w / 2) - (tracked_box.x + tracked_box.w / 2),
                    (detected_box.y + detected_box.h / 2) - (tracked_box.y + tracked_box.h / 2),
                )
                diag = hypot(tracked_box.w, tracked_box.h)
                if diag == 0:
                    continue

                if iou > 0.3 or dist < diag * 0.5:
                    score = iou + (1.0 - min(1.0, dist / diag))
                    if score > best_score:
                        best_score = score
                        best_track_index = track_index

            if best_track_index != -1:
                matched_indices[detected_index] = best_track_index
                used_tracks[best_track_index] = True

        new_tracked_boxes: list[FaceBox] = []
        new_lost_counts: list[int] = []
        new_hit_counts: list[int] = []

        for track_index, tracked_box in enumerate(self.tracked_boxes):
            matched = False
            for detected_index, matched_track_index in enumerate(matched_indices):
                if matched_track_index != track_index:
                    continue

                detected_box = detected_boxes[detected_index]
                tracked_box.x = int(tracked_box.x * (1 - self.alpha) + detected_box.x * self.alpha)
                tracked_box.y = int(tracked_box.y * (1 - self.alpha) + detected_box.y * self.alpha)
                tracked_box.w = int(tracked_box.w * (1 - self.alpha) + detected_box.w * self.alpha)
                tracked_box.h = int(tracked_box.h * (1 - self.alpha) + detected_box.h * self.alpha)
                tracked_box.label = detected_box.label
                tracked_box.confidence = detected_box.confidence
                tracked_box.raw_detection = detected_box.raw_detection

                new_tracked_boxes.append(tracked_box)
                new_lost_counts.append(0)
                new_hit_counts.append(self._hit_counts[track_index] + 1)
                matched = True
                break

            if matched:
                continue

            lost_count = self._lost_counts[track_index] + 1
            if lost_count < self.max_lost_frames:
                new_tracked_boxes.append(tracked_box)
                new_lost_counts.append(lost_count)
                new_hit_counts.append(self._hit_counts[track_index])

        for detected_index, matched_track_index in enumerate(matched_indices):
            if matched_track_index != -1:
                continue
            detected_box = detected_boxes[detected_index]
            detected_box.id = self._next_id
            self._next_id += 1
            new_tracked_boxes.append(detected_box)
            new_lost_counts.append(0)
            new_hit_counts.append(1)

        self.tracked_boxes = new_tracked_boxes
        self._lost_counts = new_lost_counts
        self._hit_counts = new_hit_counts
        return self._confirmed_tracks()

    def _confirmed_tracks(self) -> list[FaceBox]:
        return [
            tracked_box
            for index, tracked_box in enumerate(self.tracked_boxes)
            if self._hit_counts[index] >= self.min_hits
        ]
